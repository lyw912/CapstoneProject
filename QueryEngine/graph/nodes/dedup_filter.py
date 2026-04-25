"""
DedupFilter Node — Deduplication Filter

Phase 1: URL exact deduplication (comparison after normalization).
Phase 2: MinHash LSH content deduplication (datasketch, threshold 0.80).
"""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse, urlunparse

from loguru import logger

from ..state import QueryAgentState, SourceItem
from ...fusion.dedup import minhash_dedup
from ...classifiers.stance_classifier import _is_official_domain, _extract_domain


# ---------------------------------------------------------------------------
# URL Normalization
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """
    URL normalization: Remove www., query parameters, anchors, trailing slashes, and convert to lowercase.
    Used for exact deduplication.
    """
    try:
        parsed = urlparse(url.lower())
        netloc = parsed.netloc.replace("www.", "")
        path = parsed.path.rstrip("/")
        normalized = urlunparse((parsed.scheme, netloc, path, "", "", ""))
        return normalized
    except Exception:
        return url.lower()


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------

def dedup_filter_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: Deduplication filter.

    Optimization Strategy:
    1. URL exact deduplication.
    2. Content deduplication (MinHash LSH).
    3. Sorting strategy: Official domains prioritized to avoid authoritative sources being displaced by non-authoritative duplicates.
    """
    sources: List[SourceItem] = state.get("raw_sources", [])

    # Sort: Official domains first to ensure they are preserved during deduplication
    def _get_priority(s: SourceItem) -> int:
        domain = _extract_domain(s.get("url", ""))
        return 0 if _is_official_domain(domain) else 1

    sorted_sources = sorted(sources, key=_get_priority)

    # Stage 1: URL exact deduplication
    seen_urls: set = set()
    url_deduped: List[SourceItem] = []
    for s in sorted_sources:
        norm = _normalize_url(s.get("url", ""))
        if norm and norm not in seen_urls:
            seen_urls.add(norm)
            url_deduped.append(s)

    # Phase 2: MinHash LSH content deduplication (80% similarity threshold)
    content_deduped = minhash_dedup(url_deduped, threshold=0.80)

    # Restore original relative order (if needed, or keep sorted order)
    # Here we keep the sorted order because the subsequent Scorer will re-score

    trace = (
        f"[DedupFilter] Input {len(sources)} items, "
        f"After URL dedup {len(url_deduped)} items, "
        f"After content dedup {len(content_deduped)} items"
    )
    logger.info(trace)

    return {
        "deduped_sources": content_deduped,
        "trace_log": [trace],
    }
