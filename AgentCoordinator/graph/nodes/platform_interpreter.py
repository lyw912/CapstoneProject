"""
PlatformInterpreterNode: Adds platform-aware demographic interpretation to social media data.

For each platform with social data, generates a human-readable interpretation
contextualizing the stance distribution through the platform's user demographics.
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from ..state import CoordinatorState
from ...utils.platform_profiles import PLATFORM_PROFILES


async def platform_interpreter_node(state: CoordinatorState) -> dict:
    """LangGraph node: interpret per-platform stance data through demographic lens."""
    query_run = state.get("query_run")
    qa_output = query_run.get("output") if query_run else None

    if not qa_output:
        return {
            "platform_interpretations": {},
            "coordinator_trace": ["[PlatformInterpreter] No QueryAgent data — skipping"],
        }

    social = qa_output.get("social_sentiment")
    if not social or social.get("mode") != "available":
        return {
            "platform_interpretations": {},
            "coordinator_trace": ["[PlatformInterpreter] No social media data — skipping"],
        }

    per_platform = social.get("per_platform", {})
    interpretations: Dict[str, str] = {}

    for platform, stats in per_platform.items():
        if not stats:
            continue

        profile = PLATFORM_PROFILES.get(platform)
        if not profile:
            continue

        dist = stats.get("distribution", {})
        post_count = stats.get("post_count", 0) or stats.get("count", 0)
        if not dist or post_count == 0:
            continue

        dominant_stance = max(dist, key=dist.get) if dist else "neutral"
        dominant_ratio = dist.get(dominant_stance, 0)

        # Build interpretation using template
        template = profile.get("interpretation_template", "")
        platform_interp = template.format(stance=dominant_stance) if template else ""

        # Compose full interpretation
        dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in dist.items() if v > 0.05)
        interp = (
            f"**{profile['display_name']}** ({post_count} posts)\n"
            f"Stance distribution: {dist_str}\n"
            f"Dominant: {dominant_stance} ({dominant_ratio:.0%})\n"
            f"Demographic context: {profile['demographic_note']}\n"
            f"Interpretation: {platform_interp}"
        )
        interpretations[platform] = interp

    trace = (
        f"[PlatformInterpreter] Generated interpretations for "
        f"{len(interpretations)} platforms: {list(interpretations.keys())}"
    )
    logger.info(trace)

    return {
        "platform_interpretations": interpretations,
        "coordinator_trace": [trace],
    }
