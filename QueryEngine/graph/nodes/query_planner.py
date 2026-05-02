"""
QueryPlanner Node — Stance Matrix Sub-Query Generation

Receives user's raw query, uses LLM to generate a list of sub-queries covering 5 stance dimensions,
and automatically routes to appropriate search backends based on stance type.
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
# Constants
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

STANCE_MATRIX_PROMPT = """You are a senior public opinion analyst. For the following query, develop a comprehensive information gathering plan.

Query: {query}
Analysis Type: {analysis_type}

Please generate 5-8 sub-queries, must cover the following stance dimensions (at least 1 for each dimension):
1. [Official Stance official] Government/official media/corporate official statements, declarations, or policies on this matter
2. [Supporters support] Specific arguments and voices supporting/positively reviewing this event/policy/product
3. [Opponents oppose] Specific arguments and voices criticizing/opposing/questioning
4. [Neutral Analysis neutral] Objective assessments by independent analysts/research institutions/scholars
5. [Background Information background] Event causes, development process, historical context

Rules:
- Sub-queries use specific search keywords, not abstract descriptions
- Chinese topics prefer Chinese sub-queries; international topics can mix Chinese and English
- Each sub-query specifies: target_stance, target_source, priority (1-5)
- "official" stance uses "tavily", priority set to 1-2 (deep search)
- "support"/"oppose" for Chinese topics should use "mindspider_db" (social media data from Weibo/Zhihu/Bilibili), priority set to 3-4
- "support"/"oppose" for international topics use "tavily" or "any", priority set to 3-4
- "neutral"/"background" Chinese content uses "anspire", international content uses "tavily", priority set to 3-4

Output only a JSON array, no other text:
[
  {{"query": "specific search term", "target_stance": "official", "target_source": "tavily", "priority": 1}},
  ...
]"""


ANALYSIS_TYPE_PROMPT = """Determine which analysis type the following query belongs to, output only one word:
- event (breaking events/news events)
- brand (brand/product/company)
- policy (policy/regulation/rules)
- person (person/celebrity/official)
- general (general/other)

Query: {query}

Output only the type word, no other text:"""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _classify_query_type(query: str, llm: LLMClient) -> str:
    """Use LLM to determine the analysis type of the query."""
    try:
        prompt = ANALYSIS_TYPE_PROMPT.format(query=query)
        result = llm.invoke(
            system_prompt="You are a classification expert. Output exactly one English word.",
            user_prompt=prompt,
        )
        result = result.strip().lower()
        valid_types = {"event", "brand", "policy", "person", "general"}
        return result if result in valid_types else "general"
    except Exception:
        return "general"


def _parse_json_array(text: str) -> list:
    """Extract JSON array from LLM response, tolerates various formatting issues."""
    # Remove markdown code blocks
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parsing
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Extract the first [...] block
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
    Post-processing:
    1. Complete missing fields
    2. Inject official domain filtering for official stance
    3. Ensure each query has search_params field
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

        # Inject official domains for official stance (for Phase 2 Tavily include_domains)
        if item["target_stance"] == "official" and not item["search_params"].get("include_domains"):
            item["search_params"]["include_domains"] = OFFICIAL_DOMAINS_CN + OFFICIAL_DOMAINS_INTL

        enriched.append(item)

    return enriched


def _ensure_stance_coverage(sub_queries: List[SubQueryItem], query: str) -> List[SubQueryItem]:
    """Ensure coverage of at least five stances: official, support, oppose, neutral, background."""
    covered = {sq["target_stance"] for sq in sub_queries}
    required = {"official", "support", "oppose", "neutral", "background"}
    missing = required - covered

    fallback_templates = {
        "official": f"{query} official statement government response",
        "support": f"{query} support benefits advantages",
        "oppose": f"{query} opposition criticism risks",
        "neutral": f"{query} analysis assessment impact",
        "background": f"{query} background causes history",
    }

    for stance in missing:
        target = "mindspider_db" if stance in ("support", "oppose") else "any"
        sub_queries.append({
            "query": fallback_templates[stance],
            "target_stance": stance,
            "target_source": target,
            "priority": 4,
            "search_params": {},
        })

    return sub_queries


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------

async def query_planner_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: Stance matrix sub-query generation.

    Input  : state["original_query"]
    Output : sub_queries, analysis_type, search_iterations=0, max_iterations=3
    """
    query = state["original_query"]
    logger.info(f"[QueryPlanner] Starting planning: {query!r}")

    llm = _get_llm_client()

    # 1. Determine analysis type
    analysis_type = _classify_query_type(query, llm)
    logger.info(f"[QueryPlanner] Analysis type: {analysis_type}")

    # 2. LLM generates stance matrix sub-queries
    prompt = STANCE_MATRIX_PROMPT.format(query=query, analysis_type=analysis_type)
    try:
        response = llm.invoke(
            system_prompt="You are a public opinion analysis search planning expert. Output only a JSON array, no other text.",
            user_prompt=prompt,
        )
        raw_queries = _parse_json_array(response)
    except Exception as e:
        logger.error(f"[QueryPlanner] LLM call failed: {e}")
        raw_queries = []

    # 3. Post-processing: Enrich fields
    sub_queries = _enrich_sub_queries(raw_queries)

    # 4. Ensure all five stance dimensions are covered
    sub_queries = _ensure_stance_coverage(sub_queries, query)

    stances = {sq["target_stance"] for sq in sub_queries}
    trace = (
        f"[QueryPlanner] Type={analysis_type}, "
        f"Generated {len(sub_queries)} sub-queries, "
        f"Stance coverage={stances}"
    )
    logger.info(trace)

    return {
        "analysis_type": analysis_type,
        "sub_queries": sub_queries,
        "search_iterations": 0,
        "max_iterations": state.get("max_iterations", 3),
        "trace_log": [trace],
    }
