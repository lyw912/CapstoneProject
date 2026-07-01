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


def _first_sentence(text: str, max_len: int = 500) -> str:
    """Return a concise excerpt suitable for a bridged proposition."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    # Prefer the first sentence when it is informative enough.
    for sep in (". ", "。", "! ", "? "):
        if sep in cleaned:
            head, _ = cleaned.split(sep, 1)
            candidate = (head + sep.strip()).strip()
            if len(candidate) >= 40:
                return candidate[:max_len]

    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _extract_media_sections(media_text: str) -> List[str]:
    """Pull one proposition per top-level manual-report section (## title blocks)."""
    props: List[str] = []
    skip_titles = {"conclusion"}

    for block in media_text.split("\n---\n"):
        block = block.strip()
        if not block:
            continue

        section_title: Optional[str] = None
        body_parts: List[str] = []

        for line in block.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# ") and not stripped.startswith("## "):
                continue
            if stripped.startswith("## ") and section_title is None:
                section_title = stripped[3:].strip()
                if section_title.lower() in skip_titles:
                    section_title = None
                    break
                continue
            if section_title is not None:
                body_parts.append(stripped)

        if not section_title:
            continue

        excerpt = _first_sentence(" ".join(body_parts))
        if excerpt:
            props.append(f"{section_title}: {excerpt}")

    return props


def _extract_media_bullets(media_text: str) -> List[str]:
    """Legacy bullet extraction for LLM-formatted Media reports."""
    bullets: List[str] = []
    for line in media_text.split("\n"):
        line = line.strip()
        if not line.startswith("- ") or len(line) <= 20:
            continue
        content = line[2:].strip()
        if "trust:" in content or content.startswith("["):
            continue
        bullets.append(content)
    return bullets


def _bridge_media_agent(media_text: str) -> List[BridgedProposition]:
    """Extract key propositions from MediaAgent Markdown (sections and bullets)."""
    propositions: List[BridgedProposition] = []
    if not media_text:
        return propositions

    section_props = _extract_media_sections(media_text)
    bullet_props = _extract_media_bullets(media_text)

    combined: List[tuple[str, float]] = []
    seen: set[str] = set()
    candidates: List[tuple[str, float]] = [
        (text, 0.65) for text in section_props
    ] + [
        (text, 0.6) for text in bullet_props
    ]
    for item, confidence in candidates:
        key = item[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append((item, confidence))

    for content, confidence in combined[:8]:
        propositions.append({
            "prop_id": str(uuid.uuid4())[:8],
            "content": content,
            "source_agent": "media_agent",
            "stance": None,
            "confidence": confidence,
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
