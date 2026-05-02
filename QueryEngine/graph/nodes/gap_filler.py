"""
GapFiller Node

When CoverageCheck detects insufficient stance coverage, calls LLM to generate
supplementary search sub-queries (gap_queries) for missing stances, then the
graph returns to unified_search to execute the supplementary search.

LLM Model: Reuses LLMClient configured in agent.py (same as QueryPlanner).
JSON Parsing: Reuses _parse_json_array utility function from query_planner.py.

Phase 2 new node, located in the graph at coverage_check --need_more--> gap_filler --> unified_search.
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
# Official Domains (for domain filtering when supplementary searching for official stance)
# ---------------------------------------------------------------------------

OFFICIAL_DOMAINS_CN = [
    "gov.cn", "xinhua.net", "people.com.cn",
    "cctv.com", "chinadaily.com.cn", "mofcom.gov.cn",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GAP_FILL_PROMPT = """Current public opinion topic: {query}

Among collected sources, the following stances are insufficiently represented and need targeted supplementary search:
Missing stances: {missing_stances}

Please generate 1-2 specific search sub-queries for each missing stance. Requirements:
- Use specific search keywords (primarily Chinese), not abstract descriptions
- If lacking "support", search: supporter arguments / positive reviews / benefits / advantages
- If lacking "oppose", search: opposing opinions / criticism / risks / negative impacts
- If lacking "official", search: government response / official statement / regulatory attitude / policy response
- If lacking "neutral", search: expert analysis / research reports / objective assessment / third-party perspectives

Output only a JSON array, no other text:
[
  {{"query": "specific search term", "target_stance": "oppose", "target_source": "any", "priority": 2}},
  ...
]"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_json_array(text: str) -> list:
    """Extract JSON array from LLM response (tolerates various formatting issues)."""
    # Remove markdown code blocks
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
# Node Function
# ---------------------------------------------------------------------------

async def gap_filler_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: Generate supplementary search sub-queries for missing stances.

    Input: state["missing_stances"], state["original_query"]
    Output: state["gap_queries"] (list of supplementary search sub-queries, sent to unified_search)
    """
    missing: List[str] = state.get("missing_stances") or []
    query: str = state.get("original_query", "")

    if not missing:
        logger.info("[GapFiller] No missing stances, skip supplementary search")
        return {
            "gap_queries": [],
            "trace_log": ["[GapFiller] No supplementary search needed"],
        }

    prompt = GAP_FILL_PROMPT.format(
        query=query,
        missing_stances=", ".join(missing),
    )

    llm = _get_llm_client()
    try:
        response = llm.invoke(
            system_prompt="You are a search strategy expert. Output only a JSON array, no other text.",
            user_prompt=prompt,
        )
        raw_queries = _parse_json_array(response)
    except Exception as exc:
        logger.warning(f"[GapFiller] LLM call failed: {exc}")
        raw_queries = []

    # Post-processing: Complete fields + inject domain filtering for official stance
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

        # Inject official domain filtering for official stance
        if item["target_stance"] == "official" and not item["search_params"].get("include_domains"):
            item["search_params"]["include_domains"] = OFFICIAL_DOMAINS_CN

        # support/oppose prefer mindspider_db (social media data more likely to have public support/opposition voices)
        if item["target_stance"] in ("support", "oppose") and item["target_source"] == "any":
            item["target_source"] = "mindspider_db"

        processed.append(item)

    trace = (
        f"[GapFiller] Missing stances={missing}, "
        f"Generated {len(processed)} supplementary search sub-queries"
    )
    logger.info(trace)

    return {
        "gap_queries": processed,
        "trace_log": [trace],
    }
