"""
FactOpinionSeparatorNode: Separates verified facts from opinions and analytical frameworks.

Innovation 4 (Layer 3): Produces structured separation that helps readers distinguish
"what happened" from "what people think about it".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..state import CoordinatorState
from ...prompts.fact_separation_prompt import FACT_OPINION_SEPARATION_PROMPT

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _get_llm():
    from QueryEngine.llms import LLMClient
    from QueryEngine.utils.config import settings
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _parse_json_obj(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _build_all_data_summary(state: CoordinatorState) -> str:
    """Summarize all available data for the LLM prompt."""
    lines = []

    query_run = state.get("query_run")
    if query_run and query_run.get("output"):
        qa = query_run["output"]
        lines.append(f"WEB SEARCH (QueryAgent): {qa.get('total_sources_kept', 0)} sources")
        lines.append(f"Stance: {qa.get('stance_distribution', {})}")
        for cluster in qa.get("opinion_clusters", [])[:4]:
            lines.append(f"  [{cluster.get('stance')}] {cluster.get('core_argument', '')}")

    media_run = state.get("media_run")
    if media_run and media_run.get("text_output"):
        lines.append("\nMEDIA AGENT (Chinese media reporting):")
        lines.append(media_run["text_output"][:500])

    rounds = state.get("deliberation_rounds") or []
    if rounds:
        lines.append("\nDELIBERATION SYNTHESIS:")
        for r in rounds:
            if r.get("phase") == "synthesis_arbitration":
                lines.append(r.get("raw_llm_output", "")[:400])
                for cp in r.get("consensus_points", [])[:3]:
                    lines.append(f"  CONSENSUS: {cp}")
                for dp in r.get("dissent_points", [])[:3]:
                    lines.append(f"  DISSENT: {dp}")

    return "\n".join(lines)


def _build_synthesis_summary(state: CoordinatorState) -> str:
    """Extract synthesis summary from deliberation rounds."""
    rounds = state.get("deliberation_rounds") or []
    for r in rounds:
        if r.get("phase") == "synthesis_arbitration":
            return r.get("raw_llm_output", "") or ""
    consensus = state.get("deliberation_consensus") or []
    return " | ".join(consensus[:3]) if consensus else "No synthesis available."


async def fact_opinion_separator_node(state: CoordinatorState) -> dict:
    """LangGraph node: perform fact-opinion-framework separation."""
    query = state["query"]
    synthesis_summary = _build_synthesis_summary(state)
    all_data_summary = _build_all_data_summary(state)

    llm = _get_llm()

    prompt = FACT_OPINION_SEPARATION_PROMPT.format(
        query=query,
        synthesis_summary=synthesis_summary,
        all_data_summary=all_data_summary[:3000],
    )

    verified_facts: List[Dict] = []
    opinions_sentiments: List[Dict] = []
    analytical_frameworks: List[Dict] = []

    try:
        response = llm.invoke(
            system_prompt=(
                "You are a critical analyst performing Fact-Opinion Separation. "
                "Output ONLY valid JSON, no other text."
            ),
            user_prompt=prompt,
        )
        parsed = _parse_json_obj(response)
        verified_facts = parsed.get("verified_facts", [])
        opinions_sentiments = parsed.get("opinions_and_sentiments", [])
        analytical_frameworks = parsed.get("analytical_frameworks", [])
    except Exception as exc:
        logger.error(f"[FactOpinion] LLM call failed: {exc}")
        # Fallback: extract from opinion clusters
        query_run = state.get("query_run")
        if query_run and query_run.get("output"):
            qa = query_run["output"]
            for cluster in qa.get("opinion_clusters", []):
                opinions_sentiments.append({
                    "perspective": cluster.get("core_argument", ""),
                    "holders": f"{cluster.get('stance', '')} stance ({cluster.get('source_count', 0)} sources)",
                    "sentiment_intensity": "moderate",
                    "platform_distribution": {},
                    "potential_biases": [],
                })

    trace = (
        f"[FactOpinion] facts={len(verified_facts)}, "
        f"opinions={len(opinions_sentiments)}, "
        f"frameworks={len(analytical_frameworks)}"
    )
    logger.info(trace)

    return {
        "verified_facts": verified_facts,
        "opinions_sentiments": opinions_sentiments,
        "analytical_frameworks": analytical_frameworks,
        "coordinator_trace": [trace],
    }
