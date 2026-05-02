"""
GapDetector: Conditional edge router for the CRAG-driven feedback loop.

Analyzes deliberation output to determine if supplementary search is needed.
Returns one of: "sufficient" | "need_search" | "max_rounds"
"""

from __future__ import annotations

from typing import Dict, List, Literal

from loguru import logger

from ..state import CoordinatorState

MAX_SEARCH_ROUNDS = 1
MIN_PROPOSITIONS_FOR_SKIP = 3


def _extract_gaps(state: CoordinatorState) -> List[Dict]:
    """Identify information gaps from deliberation output."""
    gaps = []
    rounds = state.get("deliberation_rounds") or []
    if not rounds:
        return gaps

    # Collect data_gaps mentioned by perspectives in Phase 2.1
    for round_data in rounds:
        if round_data.get("phase") == "independent":
            for persp in round_data.get("perspectives", []):
                for gap_text in persp.get("data_gaps", []):
                    if gap_text and len(gap_text) > 10:
                        gaps.append({
                            "gap_id": f"gap_{len(gaps)}",
                            "description": gap_text,
                            "target_query": gap_text,
                            "target_source": "tavily",
                            "rationale": f"Missing for perspective: {persp.get('perspective', '')}",
                            "priority": 2,
                        })

    # Collect key unknowns from synthesis
    for round_data in rounds:
        if round_data.get("phase") == "synthesis_arbitration":
            for unknown in round_data.get("perspectives", []):  # complementary_insights
                if isinstance(unknown, str) and "unknown" in unknown.lower():
                    gaps.append({
                        "gap_id": f"gap_{len(gaps)}",
                        "description": unknown,
                        "target_query": unknown[:100],
                        "target_source": "mindspider_db",
                        "rationale": "Synthesis identified as unknown",
                        "priority": 1,
                    })

    return gaps[:3]  # Max 3 gaps to search


def gap_detector_router(
    state: CoordinatorState,
) -> Literal["sufficient", "need_search", "max_rounds"]:
    """
    Conditional edge function for LangGraph.
    Determines whether to proceed to echo_chamber or loop back for targeted search.
    """
    search_rounds = state.get("search_rounds", 0)

    if search_rounds >= MAX_SEARCH_ROUNDS:
        logger.info(f"[GapDetector] Max search rounds ({MAX_SEARCH_ROUNDS}) reached → max_rounds")
        return "max_rounds"

    bridged = state.get("bridged_propositions") or []
    if len(bridged) < MIN_PROPOSITIONS_FOR_SKIP:
        logger.info(
            f"[GapDetector] Only {len(bridged)} propositions — "
            f"insufficient data → need_search"
        )
        gaps = _extract_gaps(state)
        if gaps:
            return "need_search"

    # Check deliberation quality
    rounds = state.get("deliberation_rounds") or []
    if not rounds:
        return "sufficient"

    # If no independent analyses found or all failed
    for r in rounds:
        if r.get("phase") == "independent":
            valid = [
                p for p in r.get("perspectives", [])
                if p.get("confidence", 0) > 0.1
            ]
            if len(valid) < 2:
                logger.info("[GapDetector] Too few valid perspectives → need_search")
                return "need_search"

    logger.info("[GapDetector] Data sufficient → sufficient")
    return "sufficient"
