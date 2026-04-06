"""
OutputAssemble 节点 — 结构化输出组装

Phase 1：仅计算 stance_distribution + 汇总 sources。
Phase 2：新增 opinion_clusters（每个立场的 LLM 观点聚类）。
Phase 3：新增 knowledge_gaps + structured_summary（LLM 生成）。

Phase 2 关键变化：
  - 优先使用 classified_sources（已有 stance_label + trust_score）
  - 为每个立场生成 OpinionCluster（调用 LLM）
  - 使用 stance_coverage（来自 CoverageCheck）计算覆盖度
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from loguru import logger

from ...llms import LLMClient
from ...utils.config import settings
from ..state import OpinionCluster, QueryAgentOutput, QueryAgentState, SourceItem

# ---------------------------------------------------------------------------
# 覆盖度常量（与 coverage_check.py 保持一致）
# ---------------------------------------------------------------------------

_STANCE_THRESHOLDS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


def _compute_coverage_score(stance_counts: Dict[str, int]) -> float:
    """基于 STANCE_THRESHOLDS 计算覆盖度分数（0–1）。"""
    total_required = sum(_STANCE_THRESHOLDS.values())
    total_met = sum(
        min(stance_counts.get(s, 0), c)
        for s, c in _STANCE_THRESHOLDS.items()
    )
    return round(total_met / max(total_required, 1), 3)


# ---------------------------------------------------------------------------
# LLM 工具
# ---------------------------------------------------------------------------

OPINION_CLUSTER_PROMPT = """你是舆情分析师。针对话题"{query}"，以下是持"{stance}"立场的内容：

{sources_text}

请归纳：
1. 核心论点（1句话，简洁概括该立场的主要主张）
2. 最具代表性的原文引用（不超过100字，可直接摘录）

只输出 JSON，格式：
{{"core_argument": "...", "representative_quote": "..."}}"""


def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _parse_json_obj(text: str) -> dict:
    """从 LLM 响应中提取 JSON 对象。"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _build_sources_text(srcs: List[dict], max_sources: int = 5) -> str:
    """将来源列表转换为供 LLM 阅读的文本（最多 max_sources 条）。"""
    lines = []
    for s in srcs[:max_sources]:
        title = s.get("title", "（无标题）")
        url = s.get("url", "")
        snippet = (s.get("snippet") or "")[:200]
        lines.append(f"- [{title}]({url}): {snippet}")
    return "\n".join(lines) if lines else "（无来源）"


KNOWLEDGE_GAPS_PROMPT = """基于以下舆情分析结果，识别3-5个仍然缺失的信息维度：

话题：{query}
已覆盖立场：{covered_stances}
缺失立场：{missing_stances}
来源平台分布：{platforms}

请列出我们尚不清楚的问题，以"我们尚不清楚..."格式。
只输出JSON数组：["我们尚不清楚...", "我们尚不清楚..."]"""


async def _identify_knowledge_gaps(
    query: str,
    stance_coverage: Dict[str, int],
    missing_stances: List[str],
    sources: List[dict],
    llm: LLMClient,
) -> List[str]:
    """识别知识缺口（3-5个）"""
    if not sources:
        return ["我们尚不清楚该话题的任何信息"]

    covered = list(stance_coverage.keys())
    platforms = list(set(s.get("platform", "") for s in sources if s.get("platform")))[:10]

    prompt = KNOWLEDGE_GAPS_PROMPT.format(
        query=query,
        covered_stances="、".join(covered) if covered else "无",
        missing_stances="、".join(missing_stances) if missing_stances else "无",
        platforms="、".join(platforms) if platforms else "无",
    )

    try:
        response = llm.invoke(
            system_prompt="你是舆情分析专家。只输出JSON数组，不要有其他文字。",
            user_prompt=prompt,
        )
        gaps = _parse_json_array(response)
        return gaps[:5] if gaps else []
    except Exception as exc:
        logger.warning(f"[OutputAssemble] 知识缺口识别失败: {exc}")
        return []


def _parse_json_array(text: str) -> list:
    """从 LLM 响应中提取 JSON 数组"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


async def _generate_opinion_cluster(
    stance: str,
    srcs: List[dict],
    query: str,
    total_sources: int,
    llm: LLMClient,
) -> Optional[OpinionCluster]:
    """
    为单个立场调用 LLM 生成 OpinionCluster。
    若 LLM 调用失败，返回基于规则的最小化 cluster。
    """
    sources_text = _build_sources_text(srcs)
    prompt = OPINION_CLUSTER_PROMPT.format(
        query=query,
        stance=stance,
        sources_text=sources_text,
    )

    core_argument = ""
    representative_quote = ""

    try:
        response = llm.invoke(
            system_prompt="你是舆情分析专家。只输出 JSON，不要有其他文字。",
            user_prompt=prompt,
        )
        parsed = _parse_json_obj(response)
        core_argument = parsed.get("core_argument", "")
        representative_quote = parsed.get("representative_quote", "")
    except Exception as exc:
        logger.warning(f"[OutputAssemble] 立场 '{stance}' 聚类 LLM 失败: {exc}")
        # 降级：使用第一条来源的 snippet 作为代表性引用
        if srcs:
            core_argument = f"持{stance}立场的来源（共{len(srcs)}条）"
            representative_quote = (srcs[0].get("snippet") or "")[:100]

    if not core_argument:
        return None

    return {
        "cluster_id":            f"cluster_{stance}",
        "stance":                stance,
        "core_argument":         core_argument,
        "evidence_sources":      [s["source_id"] for s in srcs],
        "representative_quote":  representative_quote,
        "estimated_proportion":  round(len(srcs) / max(total_sources, 1), 3),
        "source_count":          len(srcs),
    }


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

async def output_assemble_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：组装最终结构化输出。

    Phase 2：
    - 优先使用 classified_sources（有 stance_label + trust_score）
    - 调用 LLM 生成每个立场的 opinion_cluster
    - knowledge_gaps / structured_summary 留 Phase 3 补全
    """
    # ------------------------------------------------------------------
    # 1. 获取来源（优先级：classified > scored > deduped > raw）
    # ------------------------------------------------------------------
    sources: List[SourceItem] = (
        state.get("classified_sources")
        or state.get("scored_sources")
        or state.get("deduped_sources")
        or state.get("raw_sources")
        or []
    )

    query = state.get("original_query", "")
    total_raw = len(state.get("raw_sources") or [])
    total_kept = len(sources)

    # ------------------------------------------------------------------
    # 2. 立场分布（Phase 2：来自真实 stance_label）
    # ------------------------------------------------------------------
    stance_counts = Counter(
        s.get("stance_label") or "unclassified" for s in sources
    )
    total = max(total_kept, 1)
    stance_distribution: Dict[str, float] = {
        stance: round(count / total, 3)
        for stance, count in stance_counts.items()
    }

    # ------------------------------------------------------------------
    # 3. 覆盖度分数
    #    优先使用 CoverageCheck 计算的 stance_coverage；否则自己计算
    # ------------------------------------------------------------------
    coverage_counts = state.get("stance_coverage") or dict(stance_counts)
    coverage_score = _compute_coverage_score(coverage_counts)

    # ------------------------------------------------------------------
    # 4. 来源排序（按 trust_score 降序）
    # ------------------------------------------------------------------
    sorted_sources = sorted(
        sources,
        key=lambda x: x.get("trust_score", 0.0),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # 5. Phase 2：OpinionCluster 生成（每个立场一个 cluster）
    # ------------------------------------------------------------------
    opinion_clusters: List[OpinionCluster] = []

    # 只为有真实来源的立场生成 cluster（排除 unclassified）
    stance_groups: Dict[str, List[dict]] = defaultdict(list)
    for s in sources:
        label = s.get("stance_label") or "unclassified"
        if label != "unclassified":
            stance_groups[label].append(s)

    if stance_groups:
        llm = _get_llm_client()
        # 按立场出现频次降序生成（保证最重要的立场优先）
        sorted_stances = sorted(
            stance_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        for stance, srcs in sorted_stances:
            cluster = await _generate_opinion_cluster(
                stance=stance,
                srcs=srcs,
                query=query,
                total_sources=total_kept,
                llm=llm,
            )
            if cluster:
                opinion_clusters.append(cluster)

    # ------------------------------------------------------------------
    # 6. 知识缺口识别
    # ------------------------------------------------------------------
    knowledge_gaps = await _identify_knowledge_gaps(
        query=query,
        stance_coverage=coverage_counts,
        missing_stances=state.get("missing_stances") or [],
        sources=sources,
        llm=llm if stance_groups else _get_llm_client(),
    )

    # ------------------------------------------------------------------
    # 7. 组装 QueryAgentOutput
    # ------------------------------------------------------------------
    output: QueryAgentOutput = {
        "original_query":     query,
        "analysis_type":      state.get("analysis_type", "general"),
        "search_iterations":  state.get("search_iterations", 0),
        "total_sources_found": total_raw,
        "total_sources_kept":  total_kept,
        "stance_distribution": stance_distribution,
        "opinion_clusters":    opinion_clusters,
        "sources":             sorted_sources,
        "knowledge_gaps":      knowledge_gaps,
        "coverage_score":      coverage_score,
        "structured_summary":  "",            # Phase 3 可选
        "trace_log":           state.get("trace_log") or [],
    }

    trace = (
        f"[OutputAssemble] 来源={total_kept}/{total_raw}, "
        f"立场分布={stance_distribution}, "
        f"SCS={coverage_score:.2f}, "
        f"clusters={len(opinion_clusters)}, "
        f"知识缺口={len(knowledge_gaps)}"
    )
    logger.info(trace)

    return {
        "query_agent_output": output,
        "trace_log": [trace],
    }
