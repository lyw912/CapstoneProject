"""
DataBridgeNode: Converts heterogeneous agent outputs into unified BridgedProposition format.

QueryAgentOutput is structured (TypedDict with sources, stance_distribution, opinion_clusters).
MediaAgent output is unstructured Markdown text.
This node bridges both into a common format for the Deliberation Engine.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from loguru import logger

from ..state import CoordinatorState, BridgedProposition


def _bridge_query_agent(query_output: Dict) -> List[BridgedProposition]:
    """Extract BridgedPropositions from QueryAgentOutput."""
    propositions: List[BridgedProposition] = []

    # From opinion_clusters (each cluster = one perspective's core argument)
    for cluster in query_output.get("opinion_clusters", []):
        stance = cluster.get("stance", "")
        argument = cluster.get("core_argument", "")
        if not argument:
            continue
        sources = query_output.get("sources", [])
        evidence_urls = [
            s.get("url", "")
            for s in sources
            if s.get("stance_label") == stance and s.get("url")
        ][:3]

        propositions.append({
            "prop_id": str(uuid.uuid4())[:8],
            "content": argument,
            "source_agent": "query_agent",
            "stance": stance,
            "confidence": cluster.get("estimated_proportion", 0.0),
            "evidence_urls": evidence_urls,
            "platform": None,
        })

    # From knowledge_gaps (treated as "we don't know" propositions)
    for gap in query_output.get("knowledge_gaps", [])[:3]:
        propositions.append({
            "prop_id": str(uuid.uuid4())[:8],
            "content": f"[KNOWLEDGE GAP] {gap}",
            "source_agent": "query_agent",
            "stance": "gap",
            "confidence": 0.5,
            "evidence_urls": [],
            "platform": None,
        })

    # Per-platform social sentiment propositions
    social = query_output.get("social_sentiment")
    if social and social.get("mode") == "available":
        per_platform = social.get("per_platform", {})
        for platform, stats in per_platform.items():
            if not stats:
                continue
            dist = stats.get("distribution", {})
            dominant = max(dist, key=dist.get) if dist else "neutral"
            ratio = dist.get(dominant, 0)
            if ratio < 0.3:
                continue
            propositions.append({
                "prop_id": str(uuid.uuid4())[:8],
                "content": (
                    f"{platform.title()} users show {dominant} sentiment "
                    f"({ratio:.0%}) on this topic"
                ),
                "source_agent": "query_agent",
                "stance": dominant,
                "confidence": ratio,
                "evidence_urls": [],
                "platform": platform,
            })

    return propositions


def _bridge_media_agent(media_text: str) -> List[BridgedProposition]:
    """Extract key propositions from MediaAgent Markdown text using heuristics."""
    propositions: List[BridgedProposition] = []
    if not media_text:
        return propositions

    lines = media_text.split("\n")
    bullet_props = []
    for line in lines:
        line = line.strip()
        if line.startswith("- ") and len(line) > 20:
            content = line[2:].strip()
            # Skip lines that look like source citations
            if "trust:" in content or content.startswith("["):
                continue
            bullet_props.append(content)

    # Take first 6 bullets as propositions
    for content in bullet_props[:6]:
        propositions.append({
            "prop_id": str(uuid.uuid4())[:8],
            "content": content,
            "source_agent": "media_agent",
            "stance": None,
            "confidence": 0.6,
            "evidence_urls": [],
            "platform": None,
        })

    return propositions


async def data_bridge_node(state: CoordinatorState) -> dict:
    """LangGraph node: bridge query_run and media_run into unified propositions."""
    query_run = state.get("query_run")
    media_run = state.get("media_run")

    all_props: List[BridgedProposition] = []

    if query_run and query_run.get("success") and query_run.get("output"):
        qa_props = _bridge_query_agent(query_run["output"])
        all_props.extend(qa_props)
        logger.info(f"[DataBridge] QueryAgent → {len(qa_props)} propositions")
    else:
        logger.warning("[DataBridge] QueryAgent output missing or failed")

    if media_run and media_run.get("success") and media_run.get("text_output"):
        ma_props = _bridge_media_agent(media_run["text_output"])
        all_props.extend(ma_props)
        logger.info(f"[DataBridge] MediaAgent → {len(ma_props)} propositions")
    else:
        logger.warning("[DataBridge] MediaAgent output missing or failed")

    trace = f"[DataBridge] Total bridged propositions: {len(all_props)}"
    logger.info(trace)

    return {
        "bridged_propositions": all_props,
        "coordinator_trace": [trace],
    }
