"""
DivergenceMatrixNode: Computes cross-source divergence matrix (Innovation 5).

For each pair of sources (query_agent, media_agent, weibo, zhihu, bilibili, etc.),
computes CSSD = 1 - cosine_similarity(stance_vector_A, stance_vector_B).

Stance vectors use [support, oppose, neutral, official, background] dimensions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..state import CoordinatorState

STANCE_DIMS = ["support", "oppose", "neutral", "official", "background"]


def _stance_vector(distribution: Dict[str, float]) -> List[float]:
    """Convert a stance distribution dict to a normalized vector."""
    vec = [distribution.get(s, 0.0) for s in STANCE_DIMS]
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    return vec


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 1.0  # treat zero vector as identical (no data)
    return dot / (mag_a * mag_b)


def _cssd(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    """Cross-Source Sentiment Difference = 1 - cosine_similarity."""
    return round(1.0 - _cosine_similarity(_stance_vector(dist_a), _stance_vector(dist_b)), 4)


def _extract_distributions(state: CoordinatorState) -> Dict[str, Dict[str, float]]:
    """Extract stance distributions from all available sources."""
    distributions: Dict[str, Dict[str, float]] = {}

    # Query Agent web search distribution
    query_run = state.get("query_run")
    if query_run and query_run.get("success") and query_run.get("output"):
        qa = query_run["output"]
        dist = qa.get("stance_distribution", {})
        if dist:
            distributions["query_agent"] = dist

        # Per-platform social media distributions
        social = qa.get("social_sentiment")
        if social and social.get("mode") == "available":
            per_platform = social.get("per_platform", {})
            for platform, stats in per_platform.items():
                if stats and stats.get("distribution"):
                    distributions[platform] = stats["distribution"]

            # Overall social media as a single source
            overall = social.get("sentiment_distribution")
            if overall:
                distributions["social_media_overall"] = overall

    # Media Agent: approximate from its sentiment mentions (heuristic)
    media_run = state.get("media_run")
    if media_run and media_run.get("success") and media_run.get("text_output"):
        text = media_run["text_output"].lower()
        # Rough heuristic: count keyword occurrences to estimate distribution
        support_kw = text.count("support") + text.count("positive") + text.count("favorable")
        oppose_kw = text.count("critic") + text.count("skeptic") + text.count("oppose")
        neutral_kw = text.count("neutral") + text.count("analytic") + text.count("objective")
        official_kw = text.count("official") + text.count("state media") + text.count("xinhua")
        total_kw = max(support_kw + oppose_kw + neutral_kw + official_kw, 1)
        distributions["media_agent"] = {
            "support": support_kw / total_kw,
            "oppose": oppose_kw / total_kw,
            "neutral": neutral_kw / total_kw,
            "official": official_kw / total_kw,
            "background": 0.0,
        }

    return distributions


async def divergence_matrix_node(state: CoordinatorState) -> dict:
    """LangGraph node: compute pairwise CSSD divergence matrix."""
    distributions = _extract_distributions(state)
    sources = list(distributions.keys())

    if len(sources) < 2:
        trace = f"[DivergenceMatrix] Only {len(sources)} source(s) — skipping matrix"
        logger.warning(trace)
        return {
            "divergence_matrix": {},
            "divergence_hotspots": [],
            "coordinator_trace": [trace],
        }

    # Compute pairwise CSSD
    matrix: Dict[str, float] = {}
    for i, src_a in enumerate(sources):
        for src_b in sources[i + 1:]:
            delta = _cssd(distributions[src_a], distributions[src_b])
            matrix[f"{src_a}|{src_b}"] = delta

    # Identify hotspots (CSSD > 0.3)
    hotspots = []
    for pair, delta in sorted(matrix.items(), key=lambda x: x[1], reverse=True):
        if delta > 0.3:
            src_a, src_b = pair.split("|")
            hotspots.append(
                f"{src_a} vs {src_b}: CSSD={delta:.3f} — notable divergence detected"
            )

    # Log summary
    if matrix:
        max_pair = max(matrix, key=matrix.get)
        max_delta = matrix[max_pair]
        min_pair = min(matrix, key=matrix.get)
        min_delta = matrix[min_pair]
        trace = (
            f"[DivergenceMatrix] {len(sources)} sources, {len(matrix)} pairs — "
            f"max divergence: {max_pair} = {max_delta:.3f}, "
            f"min divergence: {min_pair} = {min_delta:.3f}, "
            f"hotspots: {len(hotspots)}"
        )
    else:
        trace = "[DivergenceMatrix] No pairs computed"

    logger.info(trace)

    return {
        "divergence_matrix": matrix,
        "divergence_hotspots": hotspots,
        "coordinator_trace": [trace],
    }
