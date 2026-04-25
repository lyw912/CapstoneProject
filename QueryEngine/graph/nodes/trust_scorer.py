"""
TrustScorer Node

Calls compute_trust_score() for each source in deduped_sources,
writes the result to source["trust_score"], producing scored_sources.

Phase 2 new node, located in the graph at dedup_filter → trust_scorer → stance_classify.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from ...classifiers.trust_scorer import compute_trust_score
from ..state import QueryAgentState


async def trust_scorer_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: TrustScore calculation.

    Input: state["deduped_sources"]
    Output: state["scored_sources"] (each with trust_score filled)
    """
    sources: List[dict] = state.get("deduped_sources") or []

    scored: List[dict] = []
    for s in sources:
        s = dict(s)  # shallow copy, avoid modifying original objects in State
        s["trust_score"] = compute_trust_score(s)
        scored.append(s)

    if scored:
        avg = sum(s["trust_score"] for s in scored) / len(scored)
        max_s = max(s["trust_score"] for s in scored)
        min_s = min(s["trust_score"] for s in scored)
    else:
        avg = max_s = min_s = 0.0

    trace = (
        f"[TrustScorer] Processed {len(scored)} sources, "
        f"avg={avg:.3f}, max={max_s:.3f}, min={min_s:.3f}"
    )
    logger.info(trace)

    return {
        "scored_sources": scored,
        "trace_log": [trace],
    }
