"""
CoverageCheck Node

Checks coverage of each stance in classified_sources,
identifies stances that do not meet minimum requirements, and writes to missing_stances.

CoverageCheck is a synchronous node (no LLM calls) and serves as the decision basis for conditional routing.

Coverage Score Formula (SCS, Stance Coverage Score):
  SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
  K = Number of stances evaluated (4: support, oppose, official, neutral)

Phase 2 new node, located in the graph at stance_classify → coverage_check → [router].
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from loguru import logger

from ..state import QueryAgentState

# ---------------------------------------------------------------------------
# Minimum required source counts for each stance ("background" is excluded from loop checking as supplementary info)
# ---------------------------------------------------------------------------

MINIMUM_STANCE_COUNTS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


def _compute_coverage_score(stance_counts: Dict[str, int]) -> float:
    """
    Calculate stance coverage score (0–1) based on MINIMUM_STANCE_COUNTS.

    SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
    """
    total_required = sum(MINIMUM_STANCE_COUNTS.values())
    total_met = sum(
        min(stance_counts.get(s, 0), c)
        for s, c in MINIMUM_STANCE_COUNTS.items()
    )
    return round(total_met / max(total_required, 1), 3)


def coverage_check_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: Stance coverage check.

    Input: state["classified_sources"]
    Output: state["stance_coverage"] (actual count for each stance)
           state["missing_stances"] (list of stances not meeting minimum threshold)
    """
    sources: List[dict] = state.get("classified_sources") or []

    # Only count sources with explicit stance labels (exclude None / "unclassified")
    stance_counts = Counter(
        s.get("stance_label")
        for s in sources
        if s.get("stance_label") and s.get("stance_label") not in ("unclassified",)
    )

    # Identify stances that do not meet minimum thresholds
    missing: List[str] = [
        stance
        for stance, min_count in MINIMUM_STANCE_COUNTS.items()
        if stance_counts.get(stance, 0) < min_count
    ]

    coverage_score = _compute_coverage_score(dict(stance_counts))

    trace = (
        f"[CoverageCheck] Stance distribution={dict(stance_counts)}, "
        f"Missing={missing}, SCS={coverage_score:.2f}"
    )
    logger.info(trace)

    return {
        "stance_coverage":  dict(stance_counts),
        "missing_stances":  missing,
        "trace_log": [trace],
    }
