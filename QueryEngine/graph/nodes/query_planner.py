"""
QueryPlanner 节点 — 立场矩阵子查询生成

接收用户原始查询，由 LLM 生成覆盖 5 个立场维度的子查询列表，
并根据立场类型自动路由到合适的搜索后端。
"""

from __future__ import annotations

import json
import re
from typing import List

from loguru import logger

from ...llms import LLMClient
from ...utils.config import settings
from ..state import QueryAgentState, SubQueryItem


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OFFICIAL_DOMAINS_CN = [
    "gov.cn", "xinhua.net", "people.com.cn",
    "cctv.com", "chinadaily.com.cn", "mofcom.gov.cn",
    "nhc.gov.cn", "miit.gov.cn",
]

OFFICIAL_DOMAINS_INTL = [
    "gov.uk", "whitehouse.gov", "europa.eu",
    "un.org", "who.int", "imf.org",
]

STANCE_MATRIX_PROMPT = """你是一位资深舆情分析师。针对以下查询，制定全面的信息搜集计划。

查询：{query}
分析类型：{analysis_type}

请生成5-8个子查询，必须覆盖以下立场维度（每个维度至少1个）：
1.【官方立场 official】政府/官方媒体/企业官方对此事的表态、声明或政策
2.【支持方 support】支持/正面评价该事件/政策/产品的具体论据和声音
3.【反对方 oppose】批评/反对/质疑的具体论据和声音
4.【中立分析 neutral】独立分析师/研究机构/学者的客观评估
5.【背景信息 background】事件起因、发展经过、历史脉络

规则：
- 子查询使用具体搜索关键词，不是抽象描述
- 中文话题优先用中文子查询；国际话题可中英混合
- 每个子查询标注：目标立场(target_stance)、建议搜索源(target_source)、优先级(priority 1-5)
- "official" 立场使用 "tavily"，优先级设为 1-2（深度搜索）
- "support"/"oppose" 使用 "any"，优先级设为 3-4
- "neutral"/"background" 中文内容使用 "anspire"，国际内容使用 "tavily"，优先级设为 3-4

只输出 JSON 数组，不要有其他文字：
[
  {{"query": "具体搜索词", "target_stance": "official", "target_source": "tavily", "priority": 1}},
  ...
]"""


ANALYSIS_TYPE_PROMPT = """判断以下查询属于哪种分析类型，只输出一个词：
- event（突发事件/新闻事件）
- brand（品牌/产品/企业）
- policy（政策/法规/规定）
- person（人物/名人/官员）
- general（通用/其他）

查询：{query}

只输出类型词，不要其他文字："""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _classify_query_type(query: str, llm: LLMClient) -> str:
    """用 LLM 判断查询的分析类型。"""
    try:
        prompt = ANALYSIS_TYPE_PROMPT.format(query=query)
        result = llm.invoke(
            system_prompt="你是分类专家。只输出一个英文单词。",
            user_prompt=prompt,
        )
        result = result.strip().lower()
        valid_types = {"event", "brand", "policy", "person", "general"}
        return result if result in valid_types else "general"
    except Exception:
        return "general"


def _parse_json_array(text: str) -> list:
    """从 LLM 响应中提取 JSON 数组，容忍各种格式问题。"""
    # 去除 markdown 代码块
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 提取第一个 [...] 块
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def _enrich_sub_queries(sub_queries: list) -> List[SubQueryItem]:
    """
    后处理：
    1. 补全缺失字段
    2. 为 official 立场注入官方域名过滤
    3. 确保每个查询都有 search_params 字段
    """
    enriched = []
    for sq in sub_queries:
        if not isinstance(sq, dict) or "query" not in sq:
            continue

        item: SubQueryItem = {
            "query": str(sq.get("query", "")),
            "target_stance": sq.get("target_stance", "neutral"),
            "target_source": sq.get("target_source", "any"),
            "priority": int(sq.get("priority", 3)),
            "search_params": sq.get("search_params") or {},
        }

        # 为 official 立场注入官方域名（供 Phase 2 Tavily include_domains 使用）
        if item["target_stance"] == "official" and not item["search_params"].get("include_domains"):
            item["search_params"]["include_domains"] = OFFICIAL_DOMAINS_CN + OFFICIAL_DOMAINS_INTL

        enriched.append(item)

    return enriched


def _ensure_stance_coverage(sub_queries: List[SubQueryItem], query: str) -> List[SubQueryItem]:
    """确保至少覆盖 official、support、oppose、neutral、background 五个立场。"""
    covered = {sq["target_stance"] for sq in sub_queries}
    required = {"official", "support", "oppose", "neutral", "background"}
    missing = required - covered

    fallback_templates = {
        "official": f"{query} 官方声明 政府回应",
        "support": f"{query} 支持 好处 优势",
        "oppose": f"{query} 反对 质疑 风险",
        "neutral": f"{query} 分析 评估 影响",
        "background": f"{query} 背景 起因 历史",
    }

    for stance in missing:
        sub_queries.append({
            "query": fallback_templates[stance],
            "target_stance": stance,
            "target_source": "any",
            "priority": 4,
            "search_params": {},
        })

    return sub_queries


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

async def query_planner_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：立场矩阵子查询生成。

    输入  : state["original_query"]
    输出  : sub_queries, analysis_type, search_iterations=0, max_iterations=3
    """
    query = state["original_query"]
    logger.info(f"[QueryPlanner] 开始规划: {query!r}")

    llm = _get_llm_client()

    # 1. 判断分析类型
    analysis_type = _classify_query_type(query, llm)
    logger.info(f"[QueryPlanner] 分析类型: {analysis_type}")

    # 2. LLM 生成立场矩阵子查询
    prompt = STANCE_MATRIX_PROMPT.format(query=query, analysis_type=analysis_type)
    try:
        response = llm.invoke(
            system_prompt="你是舆情分析搜索规划专家。只输出JSON数组，不要有任何其他文字。",
            user_prompt=prompt,
        )
        raw_queries = _parse_json_array(response)
    except Exception as e:
        logger.error(f"[QueryPlanner] LLM 调用失败: {e}")
        raw_queries = []

    # 3. 后处理：丰富字段
    sub_queries = _enrich_sub_queries(raw_queries)

    # 4. 确保五维立场都被覆盖
    sub_queries = _ensure_stance_coverage(sub_queries, query)

    stances = {sq["target_stance"] for sq in sub_queries}
    trace = (
        f"[QueryPlanner] 类型={analysis_type}, "
        f"生成{len(sub_queries)}个子查询, "
        f"立场覆盖={stances}"
    )
    logger.info(trace)

    return {
        "analysis_type": analysis_type,
        "sub_queries": sub_queries,
        "search_iterations": 0,
        "max_iterations": state.get("max_iterations", 3),
        "trace_log": [trace],
    }
