"""
UnifiedSearchDispatcher — 统一搜索调度器（Phase 2：Tavily + Anspire）

Phase 1：只接 Tavily。
Phase 2：接入 AnspireAISearch（中文搜索增强）；InsightDB 留 Phase 2.5。
Phase 3+：接入 InsightEngine.MediaCrawlerDB（社媒数据）。

关键改动（Phase 2）：
  - _tavily_to_source_items / _anspire_to_source_items 注入 _target_stance 字段
  - 新增 _search_anspire()，路由逻辑在 _dispatch_one() 中按 target_source 分支
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
# Anspire 可选依赖（跨引擎引用，失败时降级为只用 Tavily）
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
    logger.debug(f"[Dispatcher] AnspireAISearch 不可用，中文搜索将回退到 Tavily: {_anspire_err}")

# ---------------------------------------------------------------------------
# InsightDB 可选依赖（社媒数据库）
# ---------------------------------------------------------------------------
try:
    from InsightEngine.tools.search import MediaCrawlerDB as _MediaCrawlerDB
    _INSIGHTDB_AVAILABLE = True
except Exception as _insight_err:
    _INSIGHTDB_AVAILABLE = False
    logger.debug(f"[Dispatcher] MediaCrawlerDB 不可用，社媒数据查询将跳过: {_insight_err}")


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """从 URL 中提取主域名（去 www. 前缀）。"""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _tavily_to_source_items(response: TavilyResponse, sq: Dict) -> List[Dict]:
    """将 Tavily 响应转为统一 SourceItem 格式的字典列表。"""
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
            # Phase 2：注入子查询的目标立场，供 StanceClassifier 弱标签使用
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


def _anspire_to_source_items(response, sq: Dict) -> List[Dict]:
    """将 Anspire 响应转为统一 SourceItem 格式的字典列表。"""
    items = []
    if not response:
        return items

    # AnspireAISearch 返回的结果格式
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
            # 对象属性访问
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
            # Phase 2：注入子查询的目标立场
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


def _insightdb_to_source_items(response, sq: Dict) -> List[Dict]:
    """将 InsightDB (MediaCrawlerDB) 响应转为统一 SourceItem 格式"""
    items = []
    if not response or not hasattr(response, "results"):
        return items

    for r in response.results:
        # QueryResult 对象
        url = r.url or ""
        title = r.title_or_content[:100] if r.title_or_content else ""
        snippet = r.title_or_content[:500] if r.title_or_content else ""

        items.append(dict(
            source_id=str(uuid.uuid4()),
            url=url or f"{r.platform}://{r.source_table}",  # 无URL时用平台标识
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
            rrf_score=r.hotness_score / 100.0 if r.hotness_score else 0.0,  # 归一化
            _target_stance=sq.get("target_stance", ""),
        ))
    return items


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------

class UnifiedSearchDispatcher:
    """
    统一搜索调度器。

    Phase 1：只使用 Tavily。
    Phase 2 扩展点：在 _dispatch_one() 中增加 bocha / insight_db 分支。
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
        并行调度所有子查询。

        Returns:
            (sources, errors)
        """
        tasks = [self._dispatch_one(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sources: List[Dict] = []
        errors: List[str] = []
        for sq, result in zip(sub_queries, results):
            if isinstance(result, Exception):
                msg = f"搜索失败 [{sq['query']}]: {result}"
                logger.warning(msg)
                errors.append(msg)
            elif isinstance(result, list):
                sources.extend(result)

        return sources, errors

    async def _dispatch_one(self, sq: Dict) -> List[Dict]:
        """
        对单个子查询按 target_source 路由到对应搜索后端。

        Phase 2.5 路由逻辑：
          anspire      → AnspireAISearch（中文新闻/分析）
          insight_db   → MediaCrawlerDB（社媒数据）
          tavily / any → TavilyNewsAgency
        """
        target = sq.get("target_source", "any")

        if target == "anspire" and _ANSPIRE_AVAILABLE:
            return await self._search_anspire(sq)

        if target == "insight_db" and _INSIGHTDB_AVAILABLE:
            return await self._search_insight_db(sq)

        return await self._search_tavily(sq)

    async def _search_tavily(self, sq: Dict) -> List[Dict]:
        """异步包装 Tavily 同步调用。"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_tavily_sync, sq)
        return _tavily_to_source_items(response, sq)

    def _call_tavily_sync(self, sq: Dict) -> TavilyResponse:
        """根据子查询的优先级和搜索参数选择 Tavily 工具。"""
        query = sq["query"]
        priority = sq.get("priority", 3)
        search_params = sq.get("search_params") or {}

        # include_domains 参数（用于 official 立场的域名过滤）
        # 注：TavilyNewsAgency 的 basic_search_news 接受 kwargs，透传给 Tavily API
        include_domains = search_params.get("include_domains")

        # include_domains 只在 basic_search_news 中支持，deep_search_news 不支持
        if include_domains:
            return self.tavily.basic_search_news(query, max_results=10, include_domains=include_domains)

        if priority <= 2:
            return self.tavily.deep_search_news(query)
        else:
            return self.tavily.basic_search_news(query, max_results=7)

    async def _search_anspire(self, sq: Dict) -> List[Dict]:
        """异步包装 AnspireAISearch 同步调用（中文搜索）。"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_anspire_sync, sq)
        return _anspire_to_source_items(response, sq)

    def _call_anspire_sync(self, sq: Dict):
        """调用 AnspireAISearch.comprehensive_search()。"""
        try:
            anspire = _AnspireSearch()
            return anspire.comprehensive_search(sq["query"], max_results=10)
        except Exception as exc:
            logger.warning(f"[Dispatcher] Anspire 搜索失败 [{sq['query']}]: {exc}")
            return None

    async def _search_insight_db(self, sq: Dict) -> List[Dict]:
        """异步包装 MediaCrawlerDB 同步调用（社媒数据）。"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_insightdb_sync, sq)
        return _insightdb_to_source_items(response, sq)

    def _call_insightdb_sync(self, sq: Dict):
        """调用 MediaCrawlerDB.search_topic_globally()。"""
        try:
            db = _MediaCrawlerDB()
            return db.search_topic_globally(sq["query"], limit_per_table=20)
        except Exception as exc:
            logger.warning(f"[Dispatcher] InsightDB 搜索失败 [{sq['query']}]: {exc}")
            return None
