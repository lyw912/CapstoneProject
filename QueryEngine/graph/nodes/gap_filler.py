"""
GapFiller 节点

当 CoverageCheck 发现立场覆盖不足时，调用 LLM 生成针对缺失立场的
补搜子查询（gap_queries），随后图会回到 unified_search 执行补搜。

LLM 模型：复用 agent.py 配置的 LLMClient（同 QueryPlanner 一致）。
JSON 解析：复用 query_planner.py 中的 _parse_json_array 工具函数。

Phase 2 新增节点，位于图中 coverage_check --need_more--> gap_filler --> unified_search。
"""

from __future__ import annotations

import re
import json
from typing import List

from loguru import logger

from ...llms import LLMClient
from ...utils.config import settings
from ..state import QueryAgentState, SubQueryItem

# ---------------------------------------------------------------------------
# 官方域名（用于 official 立场的补搜 domain 过滤）
# ---------------------------------------------------------------------------

OFFICIAL_DOMAINS_CN = [
    "gov.cn", "xinhua.net", "people.com.cn",
    "cctv.com", "chinadaily.com.cn", "mofcom.gov.cn",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GAP_FILL_PROMPT = """当前舆情话题：{query}

已收集的来源中，以下立场的声音不足，需要针对性补搜：
缺失立场：{missing_stances}

请为每个缺失立场生成 1-2 个具体的搜索子查询。要求：
- 使用具体的搜索关键词（中文为主），不要抽象描述
- 若缺 "support"，搜索：支持者论据 / 正面评价 / 好处 / 利好
- 若缺 "oppose"，搜索：反对意见 / 批评质疑 / 风险问题 / 负面影响
- 若缺 "official"，搜索：政府回应 / 官方声明 / 监管态度 / 政策回应
- 若缺 "neutral"，搜索：专家分析 / 研究报告 / 客观评估 / 第三方观点

只输出 JSON 数组，不要有其他文字：
[
  {{"query": "具体搜索词", "target_stance": "oppose", "target_source": "any", "priority": 2}},
  ...
]"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _parse_json_array(text: str) -> list:
    """从 LLM 响应中提取 JSON 数组（兼容各种格式问题）。"""
    # 去除 markdown 代码块
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


def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

async def gap_filler_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：为缺失立场生成补搜子查询。

    输入：state["missing_stances"]，state["original_query"]
    输出：state["gap_queries"]（补搜子查询列表，发给 unified_search）
    """
    missing: List[str] = state.get("missing_stances") or []
    query: str = state.get("original_query", "")

    if not missing:
        logger.info("[GapFiller] 无缺失立场，跳过补搜")
        return {
            "gap_queries": [],
            "trace_log": ["[GapFiller] 无需补搜"],
        }

    prompt = GAP_FILL_PROMPT.format(
        query=query,
        missing_stances="、".join(missing),
    )

    llm = _get_llm_client()
    try:
        response = llm.invoke(
            system_prompt="你是搜索策略专家。只输出 JSON 数组，不要有其他文字。",
            user_prompt=prompt,
        )
        raw_queries = _parse_json_array(response)
    except Exception as exc:
        logger.warning(f"[GapFiller] LLM 调用失败: {exc}")
        raw_queries = []

    # 后处理：补全字段 + 为 official 注入域名过滤
    processed: List[SubQueryItem] = []
    for gq in raw_queries:
        if not isinstance(gq, dict) or not gq.get("query"):
            continue

        item: SubQueryItem = {
            "query":         str(gq["query"]),
            "target_stance": gq.get("target_stance", "neutral"),
            "target_source": gq.get("target_source", "any"),
            "priority":      int(gq.get("priority", 2)),
            "search_params": gq.get("search_params") or {},
        }

        # official 立场注入官方域名过滤
        if item["target_stance"] == "official" and not item["search_params"].get("include_domains"):
            item["search_params"]["include_domains"] = OFFICIAL_DOMAINS_CN

        # support/oppose 倾向于 insight_db（社媒数据更可能有民间支持/反对声）
        if item["target_stance"] in ("support", "oppose") and item["target_source"] == "any":
            item["target_source"] = "insight_db"

        processed.append(item)

    trace = (
        f"[GapFiller] 缺失立场={missing}, "
        f"生成 {len(processed)} 个补搜子查询"
    )
    logger.info(trace)

    return {
        "gap_queries": processed,
        "trace_log": [trace],
    }
