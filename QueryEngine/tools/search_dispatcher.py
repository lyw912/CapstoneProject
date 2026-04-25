"""
UnifiedSearchDispatcher — Unified Search Dispatcher (Phase 2: Tavily + Anspire)

Phase 1: Only connected to Tavily.
Phase 2: Added AnspireAISearch (Chinese search enhancement); InsightDB reserved for Phase 2.5.
Phase 3+: Will connect to InsightEngine.MediaCrawlerDB (social media data).

Key Changes (Phase 2):
  - _tavily_to_source_items / _anspire_to_source_items inject _target_stance field
  - Added _search_anspire(), routing logic in _dispatch_one() branches by target_source
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

from .search import TavilyNewsAgency, TavilyResponse
from ..utils.config import settings

if TYPE_CHECKING:
    from ..graph.state import SourceItem, SubQueryItem

# ---------------------------------------------------------------------------
# Anspire Optional Dependency (cross-engine reference, degrades to Tavily-only on failure)
# ---------------------------------------------------------------------------
try:
    import sys, os as _os
    _proj_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from MediaEngine.tools.search import AnspireAISearch as _AnspireSearch
    _ANSPIRE_AVAILABLE = True
except Exception as _anspire_err:
    _ANSPIRE_AVAILABLE = False
    logger.debug(f"[Dispatcher] AnspireAISearch not available, Chinese search will fallback to Tavily: {_anspire_err}")

# ---------------------------------------------------------------------------
# InsightDB Optional Dependency (social media database)
# ---------------------------------------------------------------------------
try:
    from InsightEngine.tools.search import MediaCrawlerDB as _MediaCrawlerDB
    _INSIGHTDB_AVAILABLE = True
except Exception as _insight_err:
    _INSIGHTDB_AVAILABLE = False
    logger.debug(f"[Dispatcher] MediaCrawlerDB not available, social media queries will be skipped: {_insight_err}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract main domain from URL (remove www. prefix)."""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _tavily_to_source_items(response: TavilyResponse, sq: Dict) -> List[Dict]:
    """Convert Tavily response to list of dictionaries in unified SourceItem format."""
    items = []
    if not response or not response.results:
        return items

    for r in response.results:
        items.append(dict(
            source_id=str(uuid.uuid4()),
            url=r.url or "",
            title=r.title or "",
            source_api="tavily",
            platform=_extract_domain(r.url or ""),
            snippet=(r.content or "")[:500],
            full_content=r.raw_content,
            published_at=r.published_date,
            trust_score=0.0,
            stance_label=None,
            stance_confidence=0.0,
            sub_query_ref=sq["query"],
            rrf_score=r.score,
            # Phase 2: Inject target stance from sub-query for StanceClassifier weak labeling
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


def _anspire_to_source_items(response, sq: Dict) -> List[Dict]:
    """Convert Anspire response to list of dictionaries in unified SourceItem format."""
    items = []
    if not response:
        return items

    # AnspireAISearch return result format
    results = []
    if hasattr(response, "webpages"):
        results = response.webpages or []
    elif isinstance(response, dict):
        results = response.get("webpages") or []

    for r in results:
        if isinstance(r, dict):
            url = r.get("url") or ""
            title = r.get("name") or r.get("title") or ""
            snippet = r.get("snippet") or r.get("content") or ""
            published_at = r.get("date_last_crawled") or None
        else:
            # Object attribute access
            url = getattr(r, "url", "") or ""
            title = getattr(r, "name", "") or ""
            snippet = getattr(r, "snippet", "") or ""
            published_at = getattr(r, "date_last_crawled", None)

        if not url:
            continue

        items.append(dict(
            source_id=str(uuid.uuid4()),
            url=url,
            title=title,
            source_api="anspire",
            platform=_extract_domain(url),
            snippet=str(snippet)[:500],
            full_content=None,
            published_at=published_at,
            trust_score=0.0,
            stance_label=None,
            stance_confidence=0.0,
            sub_query_ref=sq["query"],
            # Phase 2: Inject target stance from sub-query
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


def _insightdb_to_source_items(response, sq: Dict) -> List[Dict]:
    """Convert InsightDB (MediaCrawlerDB) response to unified SourceItem format"""
    items = []
    if not response or not hasattr(response, "results"):
        return items

    for r in response.results:
        # QueryResult object
        url = r.url or ""
        title = r.title_or_content[:100] if r.title_or_content else ""
        snippet = r.title_or_content[:500] if r.title_or_content else ""

        items.append(dict(
            source_id=str(uuid.uuid4()),
            url=url or f"{r.platform}://{r.source_table}",  # Use platform identifier when no URL
            title=title,
            source_api="insight_db",
            platform=r.platform,
            snippet=snippet,
            full_content=r.title_or_content if len(r.title_or_content) > 500 else None,
            published_at=r.publish_time.isoformat() if r.publish_time else None,
            trust_score=0.0,
            stance_label=None,
            stance_confidence=0.0,
            sub_query_ref=sq["query"],
            rrf_score=r.hotness_score / 100.0 if r.hotness_score else 0.0,  # Normalize
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class UnifiedSearchDispatcher:
    """
    Unified Search Dispatcher.

    Phase 1: Only uses Tavily.
    Phase 2 Extension Point: Add bocha / insight_db branches in _dispatch_one().
    """

    def __init__(self):
        self._tavily: Optional[TavilyNewsAgency] = None

    @property
    def tavily(self) -> TavilyNewsAgency:
        if self._tavily is None:
            self._tavily = TavilyNewsAgency(api_key=settings.TAVILY_API_KEY)
        return self._tavily

    async def dispatch(self, sub_queries: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Dispatch all sub-queries in parallel.

        Returns:
            (sources, errors)
        """
        tasks = [self._dispatch_one(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sources: List[Dict] = []
        errors: List[str] = []
        for sq, result in zip(sub_queries, results):
            if isinstance(result, Exception):
                msg = f"Search failed [{sq['query']}]: {result}"
                logger.warning(msg)
                errors.append(msg)
            elif isinstance(result, list):
                sources.extend(result)

        return sources, errors

    async def _dispatch_one(self, sq: Dict) -> List[Dict]:
        """
        Route single sub-query to corresponding search backend by target_source.

        Phase 2.5 Routing Logic:
          anspire      → AnspireAISearch (Chinese news/analysis)
          insight_db   → MediaCrawlerDB (social media data)
          tavily / any → TavilyNewsAgency
        """
        target = sq.get("target_source", "any")

        if target == "anspire" and _ANSPIRE_AVAILABLE:
            return await self._search_anspire(sq)

        if target == "insight_db" and _INSIGHTDB_AVAILABLE:
            return await self._search_insight_db(sq)

        return await self._search_tavily(sq)

    async def _search_tavily(self, sq: Dict) -> List[Dict]:
        """Async wrapper for Tavily synchronous call."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_tavily_sync, sq)
        return _tavily_to_source_items(response, sq)

    def _call_tavily_sync(self, sq: Dict) -> TavilyResponse:
        """Select Tavily tool based on sub-query priority and search parameters."""
        query = sq["query"]
        priority = sq.get("priority", 3)
        search_params = sq.get("search_params") or {}

        # include_domains parameter (for official stance domain filtering)
        # Note: TavilyNewsAgency's basic_search_news accepts kwargs, passes through to Tavily API
        include_domains = search_params.get("include_domains")

        # include_domains only supported in basic_search_news, not in deep_search_news
        if include_domains:
            return self.tavily.basic_search_news(query, max_results=10, include_domains=include_domains)

        if priority <= 2:
            return self.tavily.deep_search_news(query)
        else:
            return self.tavily.basic_search_news(query, max_results=7)

    async def _search_anspire(self, sq: Dict) -> List[Dict]:
        """Async wrapper for AnspireAISearch synchronous call (Chinese search)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_anspire_sync, sq)
        return _anspire_to_source_items(response, sq)

    def _call_anspire_sync(self, sq: Dict):
        """Call AnspireAISearch.comprehensive_search()."""
        try:
            anspire = _AnspireSearch()
            return anspire.comprehensive_search(sq["query"], max_results=10)
        except Exception as exc:
            logger.warning(f"[Dispatcher] Anspire search failed [{sq['query']}]: {exc}")
            return None

    async def _search_insight_db(self, sq: Dict) -> List[Dict]:
        """Async wrapper for MediaCrawlerDB synchronous call (social media data)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_insightdb_sync, sq)
        return _insightdb_to_source_items(response, sq)

    def _call_insightdb_sync(self, sq: Dict):
        """Call MediaCrawlerDB.search_topic_globally()."""
        try:
            db = _MediaCrawlerDB()
            return db.search_topic_globally(sq["query"], limit_per_table=20)
        except Exception as exc:
            logger.warning(f"[Dispatcher] InsightDB search failed [{sq['query']}]: {exc}")
            return None
