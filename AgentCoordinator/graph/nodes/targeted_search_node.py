"""
TargetedSearchNode: CRAG-driven supplementary search (Innovation 2).

When GapDetector identifies information gaps, this node performs targeted search
via the fastest available channel:
  1. MindSpiderDB (~100ms) — for social media data gaps
  2. Tavily API (~2-5s) — for web search data gaps
  3. BroadTopicExtraction background trigger (~30s) — for complete data absence
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..state import CoordinatorState, SearchGap

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def _search_mindspider(keyword: str, limit: int = 5) -> List[Dict]:
    """Query MindSpiderDB for social media data."""
    try:
        from QueryEngine.tools.mindspider_search import MindSpiderDB
        db = MindSpiderDB()
        resp = db.search_topic_globally(keyword, limit_per_table=limit)
        results = []
        for item in (resp.results if hasattr(resp, "results") else []):
            results.append({
                "title": getattr(item, "title", "") or getattr(item, "content", "")[:50],
                "snippet": (getattr(item, "content", "") or "")[:200],
                "url": getattr(item, "url", "") or getattr(item, "note_url", ""),
                "source": "mindspider_db",
                "platform": getattr(item, "platform", ""),
            })
        return results
    except Exception as exc:
        logger.warning(f"[TargetedSearch] MindSpiderDB search failed: {exc}")
        return []


async def _search_tavily(query: str, max_results: int = 5) -> List[Dict]:
    """Search via Tavily API."""
    try:
        from QueryEngine.utils.config import settings
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        resp = client.search(query=query, max_results=max_results, search_depth="basic")
        results = []
        for r in resp.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("content", "")[:200],
                "url": r.get("url", ""),
                "source": "tavily",
                "platform": "web",
            })
        return results
    except Exception as exc:
        logger.warning(f"[TargetedSearch] Tavily search failed: {exc}")
        return []


def _detect_gaps(state: CoordinatorState) -> List[SearchGap]:
    """Extract gaps from deliberation state."""
    gaps: List[SearchGap] = []
    rounds = state.get("deliberation_rounds") or []

    for round_data in rounds:
        if round_data.get("phase") == "independent":
            for persp in round_data.get("perspectives", []):
                for gap_text in persp.get("data_gaps", []):
                    if gap_text and len(gap_text) > 10:
                        gaps.append({
                            "gap_id": f"gap_{len(gaps)}",
                            "description": gap_text,
                            "target_query": gap_text[:100],
                            "target_source": "tavily",
                            "rationale": f"Perspective '{persp.get('perspective', '')}' identified gap",
                            "priority": 2,
                        })

    return gaps[:2]  # Only handle top 2 gaps


async def targeted_search_node(state: CoordinatorState) -> dict:
    """LangGraph node: perform targeted supplementary search for identified gaps."""
    search_rounds = state.get("search_rounds", 0) + 1
    t0 = time.time()

    gaps = _detect_gaps(state)
    if not gaps:
        logger.info("[TargetedSearch] No gaps detected — skipping")
        return {
            "search_rounds": search_rounds,
            "search_gaps": [],
            "supplementary_results": [],
            "coordinator_trace": [f"[TargetedSearch] No gaps to fill (round {search_rounds})"],
        }

    logger.info(f"[TargetedSearch] Round {search_rounds}: searching {len(gaps)} gaps")

    all_results: List[Dict] = []
    for gap in gaps:
        source = gap.get("target_source", "tavily")
        query = gap.get("target_query", "")
        if not query:
            continue

        if source == "mindspider_db":
            results = await _search_mindspider(query)
        else:
            results = await _search_tavily(query)

        logger.info(
            f"[TargetedSearch] Gap '{gap['description'][:50]}' via {source}: {len(results)} results"
        )
        all_results.extend(results)

    duration = time.time() - t0
    trace = (
        f"[TargetedSearch] Round {search_rounds}: {len(gaps)} gaps → "
        f"{len(all_results)} new results in {duration:.1f}s"
    )
    logger.info(trace)

    return {
        "search_rounds": search_rounds,
        "search_gaps": gaps,
        "supplementary_results": all_results,
        "coordinator_trace": [trace],
    }
