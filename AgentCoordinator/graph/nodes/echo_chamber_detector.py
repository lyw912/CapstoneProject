"""
EchoChamberDetectorNode: Detects echo chambers and silent majority signals (Innovation 4).

Computes Stance Entropy per platform and identifies where a single stance dominates.
Also detects silent majority signals: when web search shows opposition not visible in social media.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from loguru import logger

from ..state import CoordinatorState

ENTROPY_WARNING_THRESHOLD = 0.5   # low entropy = echo chamber risk
MIN_POSTS_FOR_ANALYSIS = 3


def _shannon_entropy(distribution: Dict[str, float]) -> float:
    """Shannon entropy: H = -Σ p log p"""
    entropy = 0.0
    for p in distribution.values():
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _detect_per_platform_echo_chambers(qa_output: Optional[Dict]) -> List[str]:
    """Check per-platform stance distributions for echo chamber signals."""
    warnings = []
    if not qa_output:
        return warnings

    social = qa_output.get("social_sentiment")
    if not social or social.get("mode") != "available":
        return warnings

    per_platform = social.get("per_platform", {})
    for platform, stats in per_platform.items():
        if not stats:
            continue
        dist = stats.get("distribution", {})
        post_count = stats.get("post_count", 0)
        if not dist or post_count < MIN_POSTS_FOR_ANALYSIS:
            continue

        entropy = _shannon_entropy(dist)
        dominant_stance = max(dist, key=dist.get) if dist else None
        dominant_ratio = dist.get(dominant_stance, 0) if dominant_stance else 0

        if entropy < ENTROPY_WARNING_THRESHOLD and dominant_ratio > 0.7:
            warnings.append(
                f"{platform}: stance entropy={entropy:.3f} (low), "
                f"dominant={dominant_stance} ({dominant_ratio:.0%}) — "
                f"possible echo chamber effect"
            )

    # Check overall social sentiment
    overall_dist = social.get("sentiment_distribution", {})
    if overall_dist:
        entropy = _shannon_entropy(overall_dist)
        dominant = max(overall_dist, key=overall_dist.get) if overall_dist else None
        if dominant and entropy < ENTROPY_WARNING_THRESHOLD and overall_dist.get(dominant, 0) > 0.7:
            warnings.append(
                f"Overall social media: entropy={entropy:.3f}, "
                f"dominant={dominant} ({overall_dist.get(dominant, 0):.0%}) — "
                f"overall social discourse may be homogeneous"
            )

    # Check diversity score from Phase 3.6
    diversity = social.get("content_diversity", 1.0)
    if diversity < 0.7:
        warnings.append(
            f"Content diversity score={diversity:.2f} (<0.7) — "
            f"results may be affected by coordinated posting"
        )

    return warnings


def _detect_silent_majority(qa_output: Optional[Dict]) -> Optional[str]:
    """
    Detect silent majority hypothesis:
    Web search shows significant opposition, but social media is overwhelmingly supportive.
    """
    if not qa_output:
        return None

    web_dist = qa_output.get("stance_distribution", {})
    social = qa_output.get("social_sentiment")

    if not social or social.get("mode") != "available":
        return None

    social_dist = social.get("sentiment_distribution", {})
    if not web_dist or not social_dist:
        return None

    web_oppose = web_dist.get("oppose", 0)
    social_oppose = social_dist.get("oppose", 0)
    social_support = social_dist.get("support", 0) + social_dist.get("neutral", 0)

    # Signal: web has notable opposition, social media doesn't
    if web_oppose > 0.2 and social_oppose < 0.1 and social_support > 0.7:
        web_pct = f"{web_oppose:.0%}"
        social_pct = f"{social_oppose:.0%}"
        return (
            f"Silent Majority Hypothesis: Web search shows {web_pct} opposition-leaning sources, "
            f"but social media shows only {social_pct} opposition. "
            f"This divergence may indicate: (1) opposition voices are present but not socially amplified; "
            f"(2) platform algorithmic suppression; "
            f"(3) social desirability bias in social media posting."
        )

    return None


async def echo_chamber_detector_node(state: CoordinatorState) -> dict:
    """LangGraph node: compute echo chamber indicators and silent majority signals."""
    query_run = state.get("query_run")
    qa_output = query_run.get("output") if query_run else None

    warnings = _detect_per_platform_echo_chambers(qa_output)
    silent_hypothesis = _detect_silent_majority(qa_output)

    if silent_hypothesis:
        warnings.append(f"[SILENT MAJORITY] {silent_hypothesis}")

    # Also flag if only one agent has data
    if not state.get("query_run", {}) or not state.get("query_run", {}).get("success"):
        warnings.append("Web search data unavailable — analysis based only on Media Agent")
    if not state.get("media_run", {}) or not state.get("media_run", {}).get("success"):
        warnings.append("Media Agent data unavailable — no Chinese media perspective included")

    trace = (
        f"[EchoChamber] {len(warnings)} warning(s) detected"
        + (f"; silent_majority_hypothesis generated" if silent_hypothesis else "")
    )
    logger.info(trace)

    return {
        "echo_warnings": warnings,
        "silent_majority_hypothesis": silent_hypothesis,
        "coordinator_trace": [trace],
    }
