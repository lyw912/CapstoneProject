"""
Evaluation metrics implementation

Reference: Architecture Doc v2.0 Part 3 § 12

Metrics System:
  SCS  Stance Coverage Score    — Core metric, measures if all stances are adequately covered
  SDI  Source Diversity Index   — Shannon entropy normalized, measures platform diversity
  SBS  Stance Balance Score     — 1 - Gini coefficient, measures stance distribution balance
  TSM  Trust Score Mean         — Average trust score
  E2E  End-to-End Latency       — End-to-end latency (passed by caller)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional

# Phase 2 target thresholds
PHASE2_TARGETS = {
    "scs": 0.70,
    "sdi": 0.50,
    "sbs": 0.50,
    "tsm": 0.50,
    "e2e": 180.0,
}

# Minimum thresholds for stance coverage calculation (consistent with coverage_check.py)
_STANCE_THRESHOLDS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


# ---------------------------------------------------------------------------
# SCS — Stance Coverage Score
# ---------------------------------------------------------------------------

def stance_coverage_score(output: dict) -> float:
    """
    Stance coverage score (0–1).

    SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)

    Target: ≥ 0.70 (Phase 2), ≥ 0.75 (Phase 3)
    """
    sources: List[dict] = output.get("sources") or []
    stance_counts = Counter(
        s.get("stance_label")
        for s in sources
        if s.get("stance_label") and s["stance_label"] not in ("unclassified",)
    )

    if not stance_counts:
        return 0.0

    scores = []
    for stance, threshold in _STANCE_THRESHOLDS.items():
        actual = stance_counts.get(stance, 0)
        scores.append(min(actual / threshold, 1.0))

    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# SDI — Source Diversity Index
# ---------------------------------------------------------------------------

def source_diversity_index(output: dict) -> float:
    """
    Source diversity index (0–1), based on normalized Shannon entropy.

    SDI = H(platforms) / log2(|unique_platforms|)

    Target: ≥ 0.50 (Phase 2)
    """
    sources: List[dict] = output.get("sources") or []
    if not sources:
        return 0.0

    platform_counts = Counter(s.get("platform", "") or "unknown" for s in sources)
    n_platforms = len(platform_counts)

    if n_platforms <= 1:
        return 0.0

    total = sum(platform_counts.values())
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in platform_counts.values()
        if c > 0
    )
    max_entropy = math.log2(n_platforms)
    return round(entropy / max_entropy if max_entropy > 0 else 0.0, 3)


# ---------------------------------------------------------------------------
# SBS — Stance Balance Score
# ---------------------------------------------------------------------------

def stance_balance_score(output: dict) -> float:
    """
    Stance balance score (0–1), based on 1 - Gini coefficient.

    Perfect balance → SBS = 1.0; Skewed to one stance → SBS → 0.

    Target: ≥ 0.50 (Phase 2)
    """
    dist: Dict[str, float] = output.get("stance_distribution") or {}
    # Filter out unclassified
    values = [v for k, v in dist.items() if k != "unclassified"]

    if not values or len(values) <= 1:
        return 0.0

    n = len(values)
    mean_v = sum(values) / n
    if mean_v == 0:
        return 0.0

    # Gini = Σ|v_i - v_j| / (2 * n^2 * mean)
    gini = sum(abs(v1 - v2) for v1 in values for v2 in values) / (2 * n * n * mean_v)
    return round(max(0.0, 1.0 - gini), 3)


# ---------------------------------------------------------------------------
# TSM — Trust Score Mean
# ---------------------------------------------------------------------------

def trust_score_mean(output: dict) -> float:
    """
    Average trust score (0–1).

    Target: ≥ 0.50 (Phase 2)
    """
    sources: List[dict] = output.get("sources") or []
    if not sources:
        return 0.0

    scores = [float(s.get("trust_score") or 0.0) for s in sources]
    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# Comprehensive Calculation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    output: dict,
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.

    Args:
        output:          QueryAgentOutput dictionary
        elapsed_seconds: End-to-end elapsed time (seconds), None if not measured

    Returns:
        Dictionary containing scs/sdi/sbs/tsm/e2e
    """
    return {
        "scs": stance_coverage_score(output),
        "sdi": source_diversity_index(output),
        "sbs": stance_balance_score(output),
        "tsm": trust_score_mean(output),
        "e2e": round(elapsed_seconds, 1) if elapsed_seconds is not None else -1.0,
        "sources_count":    output.get("total_sources_kept", 0),
        "search_iterations": output.get("search_iterations", 0),
        "coverage_score":   output.get("coverage_score", 0.0),
    }


def check_phase2_pass(metrics: Dict[str, float]) -> bool:
    """
    Check if Phase 2 acceptance criteria are met:
      SCS ≥ 0.70, SDI ≥ 0.50, E2E < 180s
    """
    e2e = metrics.get("e2e", -1.0)
    return (
        metrics.get("scs", 0) >= PHASE2_TARGETS["scs"]
        and metrics.get("sdi", 0) >= PHASE2_TARGETS["sdi"]
        and (e2e < 0 or e2e < PHASE2_TARGETS["e2e"])
    )


def format_metrics_report(query: str, metrics: Dict[str, float]) -> str:
    """Format evaluation results for a single query."""
    passed = check_phase2_pass(metrics)
    status = "✅ PASS" if passed else "❌ FAIL"

    lines = [
        f"查询: {query}",
        f"  SCS={metrics.get('scs', 0):.3f}  (目标≥0.70)",
        f"  SDI={metrics.get('sdi', 0):.3f}  (目标≥0.50)",
        f"  SBS={metrics.get('sbs', 0):.3f}  (目标≥0.50)",
        f"  TSM={metrics.get('tsm', 0):.3f}  (目标≥0.50)",
        f"  E2E={metrics.get('e2e', -1):.1f}s (目标<180s)",
        f"  来源数={metrics.get('sources_count', 0)}, "
        f"搜索轮次={metrics.get('search_iterations', 0)}",
        f"  结论: {status}",
    ]
    return "\n".join(lines)
