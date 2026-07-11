"""Provider-normalizing source gateway.

The gateway does not hide missing keys or failed network calls. Every provider
route produces a diagnostic. Local replay data is used only as an explicit,
diagnosed fallback so the engine can still run end to end without external API
credentials.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from AgentCoordinator.utils.platform_profiles import canonical_social_platform

from ..contracts import (
    NormalizedItem,
    ProviderDiagnostic,
    RetrievalTask,
    RetrievalTaskResult,
    utc_now,
)


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower() or "local"
    except Exception:
        return "local"


def _platform_key(url: str, fallback: str = "web") -> str:
    fallback_key = str(fallback or "").strip().lower().replace("www.", "")
    social_key = canonical_social_platform(fallback_key) or canonical_social_platform(_domain(url))
    return social_key or fallback_key or _domain(url) or "web"


def _language(text: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "mixed" if any(char.isascii() and char.isalpha() for char in text) else "zh"
    return "en"


class SourceGateway:
    """Normalize configured search providers into NormalizedItem objects."""

    def __init__(self, settings: Optional[Any] = None):
        if settings is None:
            try:
                from config import settings as loaded_settings
            except Exception:
                loaded_settings = None
            settings = loaded_settings
        self.settings = settings
        self.diagnostics: List[ProviderDiagnostic] = []

    def search_many(self, tasks: List[RetrievalTask]) -> Tuple[List[NormalizedItem], List[RetrievalTaskResult], List[ProviderDiagnostic]]:
        self.diagnostics = []
        all_items: List[NormalizedItem] = []
        results: List[RetrievalTaskResult] = []

        for task in tasks:
            if task.target_source == "mindspider_db":
                started = time.time()
                if self._mindspider_enabled():
                    provider, raw_items, errors = self._search_mindspider(task)
                else:
                    provider, raw_items, errors = self._mindspider_disabled(task)
                elapsed_ms = int((time.time() - started) * 1000)
                normalized = [
                    self._to_normalized(raw, task, index, provider=provider)
                    for index, raw in enumerate(raw_items)
                    if (raw.get("text") or raw.get("title") or raw.get("url"))
                ]
                all_items.extend(normalized)
                results.append(
                    RetrievalTaskResult(
                        task_id=task.task_id,
                        provider=provider,
                        status="ok" if normalized and not errors else "error" if errors else "empty",
                        items_returned=len(normalized),
                        errors=errors,
                        elapsed_ms=elapsed_ms,
                    )
                )
                continue

            started = time.time()
            provider, raw_items, errors = self._search_task(task)
            elapsed_ms = int((time.time() - started) * 1000)
            normalized = [
                self._to_normalized(raw, task, index, provider=provider)
                for index, raw in enumerate(raw_items)
                if (raw.get("text") or raw.get("title") or raw.get("url"))
            ]
            all_items.extend(normalized)
            results.append(
                RetrievalTaskResult(
                    task_id=task.task_id,
                    provider=provider,
                    status="ok" if normalized and not errors else "error" if errors else "empty",
                    items_returned=len(normalized),
                    errors=errors,
                    elapsed_ms=elapsed_ms,
                )
            )

        if not all_items and self._allow_fixture_fallback():
            fixture_items = self._fixture_items(tasks[0] if tasks else None)
            all_items.extend(fixture_items)
            results.append(
                RetrievalTaskResult(
                    task_id="local_fixture",
                    provider="local_fixture",
                    status="ok",
                    items_returned=len(fixture_items),
                    errors=[],
                    elapsed_ms=0,
                )
            )
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="local_fixture",
                    capability="replay_source_acquisition",
                    status="used",
                    route="fixture",
                    configured=True,
                    warnings=[
                        "External source acquisition was unavailable or empty; local replay data was used for end-to-end execution."
                    ],
                    metadata={"items": len(fixture_items)},
                )
            )

        return all_items, results, list(self.diagnostics)

    @staticmethod
    def _task_queries(task: RetrievalTask, default_limit: int) -> List[str]:
        max_api_calls = task.budget.get("max_api_calls") if isinstance(task.budget, dict) else None
        try:
            limit = int(max_api_calls or default_limit)
        except Exception:
            limit = default_limit
        limit = max(1, min(default_limit, limit))
        queries: List[str] = []
        for query in [task.query, *(task.query_variants or [])]:
            text = " ".join(str(query or "").split())
            key = text.lower()
            if text and key not in {item.lower() for item in queries}:
                queries.append(text)
        return queries[:limit]

    @staticmethod
    def _dedupe_raw_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in items:
            url = str(item.get("url") or "").strip()
            platform = str(item.get("platform") or "").strip().lower()
            text = " ".join(str(item.get("text") or item.get("title") or "").split())
            key = url.split("#")[0] if url else f"{platform}:{text[:240]}"
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _search_task(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        provider = self._selected_provider()
        if provider in {"bocha", "anspire", "tavily"} and not self._provider_configured(provider):
            required_key = {
                "bocha": "BOCHA_WEB_SEARCH_API_KEY",
                "anspire": "ANSPIRE_API_KEY",
                "tavily": "TAVILY_API_KEY",
            }[provider]
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider=provider,
                    capability="query_time_search",
                    status="not_configured",
                    route=f"SEARCH_TOOL_TYPE={self._search_tool_label()}",
                    configured=False,
                    required=False,
                    warnings=[f"{provider} source acquisition is disabled because {required_key} is not set."],
                )
            )
            return provider, [], []
        if provider in {"bocha", "anspire", "tavily"}:
            all_items: List[Dict[str, Any]] = []
            all_errors: List[str] = []
            queries = self._task_queries(task, default_limit=3 if task.target_source == "web" else 2)
            for query in queries:
                variant_task = replace(task, query=query)
                if provider == "bocha":
                    _provider, items, errors = self._search_bocha(variant_task)
                elif provider == "anspire":
                    _provider, items, errors = self._search_anspire(variant_task)
                else:
                    _provider, items, errors = self._search_tavily(variant_task)
                for item in items:
                    item["retrieval_query"] = query
                all_items.extend(items)
                all_errors.extend(errors)
            return provider, self._dedupe_raw_items(all_items), all_errors

        self.diagnostics.append(
            ProviderDiagnostic(
                provider="web_search",
                capability="query_time_search",
                status="not_configured",
                route="none",
                configured=False,
                required=False,
                warnings=["No Tavily, Bocha, or Anspire search key is configured; external source acquisition is disabled."],
            )
        )
        return "none", [], []

    def _mindspider_enabled(self) -> bool:
        if not self.settings:
            return False
        return bool(getattr(self.settings, "COORDINATOR_ENABLE_MINDSPIDER_DB", False))

    def _load_mindspider_db(self):
        from QueryEngine.tools.mindspider_search import MindSpiderDB

        return MindSpiderDB

    def _mindspider_disabled(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        self.diagnostics.append(
            ProviderDiagnostic(
                provider="mindspider_db",
                capability="social_source_acquisition",
                status="not_configured",
                route="COORDINATOR_ENABLE_MINDSPIDER_DB=false",
                configured=False,
                required=False,
                warnings=["MindSpiderDB platform acquisition is disabled; web evidence can still run end to end."],
                metadata={"task_id": task.task_id, "query": task.query},
            )
        )
        return "mindspider_db", [], []

    def _search_mindspider(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        route = "COORDINATOR_ENABLE_MINDSPIDER_DB=true"
        queries = self._task_queries(task, default_limit=5)
        self.diagnostics.append(
            ProviderDiagnostic(
                provider="mindspider_db",
                capability="social_source_acquisition",
                status="configured",
                route=route,
                configured=True,
                metadata={"task_id": task.task_id, "query": task.query, "queries": queries},
            )
        )
        try:
            db_class = self._load_mindspider_db()
            db = db_class()
            # Fail fast on unreachable credentials instead of returning a misleading
            # "empty" platform sample after table-level exceptions are swallowed.
            with db.engine.connect():
                pass
            per_query_limit = max(1, min(int(task.max_results or 8), int(getattr(self.settings, "DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE", 20) or 20)))
            items: List[Dict[str, Any]] = []
            comment_items = 0
            for query_index, query in enumerate(queries):
                response = db.search_topic_globally(query, limit_per_table=per_query_limit)
                rows = list(getattr(response, "results", []) or [])
                for row in rows:
                    raw = self._mindspider_row_to_raw(row)
                    raw["retrieval_query"] = query
                    items.append(raw)

                # Comments often contain the user reaction while the parent post title
                # carries only a broad topic. Limit this to the first two high-signal
                # queries to avoid broad full-table scans on every run.
                if query_index < 2 and hasattr(db, "search_comments"):
                    comments = list(db.search_comments(query, limit_per_table=max(1, min(3, per_query_limit))) or [])
                    for comment in comments:
                        raw = self._mindspider_comment_to_raw(comment)
                        raw["retrieval_query"] = query
                        items.append(raw)
                        comment_items += 1

            items = self._dedupe_raw_items(items)
            platforms: Dict[str, int] = {}
            for item in items:
                platform = str(item.get("platform") or "unknown")
                platforms[platform] = platforms.get(platform, 0) + 1
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="mindspider_db",
                    capability="social_source_acquisition",
                    status="used" if items else "empty",
                    route=route,
                    configured=True,
                    warnings=[] if items else ["MindSpiderDB was reachable but returned no matching platform samples for this query."],
                    metadata={"items": len(items), "comment_items": comment_items, "platforms": platforms, "task_id": task.task_id, "queries": queries},
                )
            )
            return "mindspider_db", items, []
        except Exception as exc:
            error = str(exc)
            logger.warning("[SourceGateway] MindSpiderDB search failed for {}: {}", task.query, error)
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="mindspider_db",
                    capability="social_source_acquisition",
                    status="error",
                    route=route,
                    configured=True,
                    errors=[error],
                    metadata={"task_id": task.task_id, "query": task.query},
                )
            )
            return "mindspider_db", [], [error]

    @staticmethod
    def _mindspider_comment_to_raw(comment: Any) -> Dict[str, Any]:
        platform = str(getattr(comment, "platform", "") or "mindspider")
        text = " ".join(str(getattr(comment, "content", "") or "").split())
        digest = hashlib.sha1(f"{platform}:{text}".encode("utf-8", errors="ignore")).hexdigest()[:12]
        published_at = getattr(comment, "publish_time", None)
        if hasattr(published_at, "isoformat"):
            published_at = published_at.isoformat()
        like_count = getattr(comment, "like_count", None)
        return {
            "url": f"mindspider://{platform}/comment/{digest}",
            "title": text[:100],
            "text": text,
            "published_at": published_at,
            "source_type": "comment",
            "platform": platform,
            "source_name": platform,
            "acquisition_source": "mindspider_db",
            "engagement": {"likes": like_count} if like_count is not None else {},
        }

    @staticmethod
    def _mindspider_row_to_raw(row: Any) -> Dict[str, Any]:
        platform = str(getattr(row, "platform", "") or "mindspider")
        table = str(getattr(row, "source_table", "") or "mindspider")
        text = " ".join(str(getattr(row, "title_or_content", "") or "").split())
        url = str(getattr(row, "url", "") or "")
        if not url:
            url = f"mindspider://{platform}/{table}/{hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
        published_at = getattr(row, "publish_time", None)
        if hasattr(published_at, "isoformat"):
            published_at = published_at.isoformat()
        return {
            "url": url,
            "title": text[:100],
            "text": text,
            "published_at": published_at,
            "source_type": "ugc",
            "platform": platform,
            "source_name": platform,
            "acquisition_source": "mindspider_db",
        }

    def _selected_provider(self) -> str:
        if not self.settings:
            return "none"
        search_tool = str(getattr(self.settings, "SEARCH_TOOL_TYPE", "") or "").lower()
        configured_choice = {
            "bochaapi": "bocha",
            "anspireapi": "anspire",
            "tavilyapi": "tavily",
        }.get(search_tool)
        if configured_choice:
            return configured_choice
        if getattr(self.settings, "TAVILY_API_KEY", None):
            return "tavily"
        if getattr(self.settings, "BOCHA_WEB_SEARCH_API_KEY", None):
            return "bocha"
        if getattr(self.settings, "ANSPIRE_API_KEY", None):
            return "anspire"
        return "none"

    def _provider_configured(self, provider: str) -> bool:
        if provider == "bocha":
            return bool(getattr(self.settings, "BOCHA_WEB_SEARCH_API_KEY", None))
        if provider == "anspire":
            return bool(getattr(self.settings, "ANSPIRE_API_KEY", None))
        if provider == "tavily":
            return bool(getattr(self.settings, "TAVILY_API_KEY", None))
        return False

    def _search_tool_label(self) -> str:
        return str(getattr(self.settings, "SEARCH_TOOL_TYPE", "") or "auto")

    def _search_anspire(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        api_key = getattr(self.settings, "ANSPIRE_API_KEY", None)
        base_url = getattr(self.settings, "ANSPIRE_BASE_URL", None) or "https://plugin.anspire.cn/api/ntsearch/search"
        timeout = int(getattr(self.settings, "SEARCH_TIMEOUT", 30) or 30)
        self.diagnostics.append(
            ProviderDiagnostic(
                provider="anspire",
                capability="query_time_search",
                status="configured",
                route=base_url,
                configured=True,
                metadata={"task_id": task.task_id, "query": task.query},
            )
        )
        try:
            response = requests.get(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                params={"query": task.query, "top_k": task.max_results},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results") or payload.get("webpages") or []
            items = [
                {
                    "url": row.get("url") or "",
                    "title": row.get("title") or row.get("name") or "",
                    "text": row.get("content") or row.get("snippet") or "",
                    "published_at": row.get("date") or row.get("date_last_crawled"),
                    "source_type": "search_result",
                    "platform": _platform_key(row.get("url") or ""),
                    "source_name": _domain(row.get("url") or ""),
                    "acquisition_source": "anspire",
                }
                for row in rows
                if isinstance(row, dict)
            ]
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="anspire",
                    capability="query_time_search",
                    status="used",
                    route=base_url,
                    configured=True,
                    metadata={"items": len(items), "task_id": task.task_id},
                )
            )
            return "anspire", items, []
        except Exception as exc:
            error = str(exc)
            logger.warning("[SourceGateway] Anspire search failed for {}: {}", task.query, error)
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="anspire",
                    capability="query_time_search",
                    status="error",
                    route=base_url,
                    configured=True,
                    errors=[error],
                    metadata={"task_id": task.task_id, "query": task.query},
                )
            )
            return "anspire", [], [error]

    def _search_bocha(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        api_key = getattr(self.settings, "BOCHA_WEB_SEARCH_API_KEY", None)
        base_url = getattr(self.settings, "BOCHA_BASE_URL", None) or "https://api.bocha.cn/v1/ai-search"
        timeout = int(getattr(self.settings, "SEARCH_TIMEOUT", 30) or 30)
        self.diagnostics.append(
            ProviderDiagnostic(
                provider="bocha",
                capability="query_time_search",
                status="configured",
                route=base_url,
                configured=True,
                metadata={"task_id": task.task_id, "query": task.query},
            )
        )
        try:
            response = requests.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": task.query, "count": task.max_results},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            rows = self._bocha_rows(payload)
            items = [
                {
                    "url": row.get("url") or row.get("link") or "",
                    "title": row.get("name") or row.get("title") or "",
                    "text": row.get("snippet") or row.get("summary") or row.get("content") or "",
                    "published_at": row.get("datePublished") or row.get("date") or row.get("published_at"),
                    "source_type": "search_result",
                    "platform": _platform_key(row.get("url") or row.get("link") or ""),
                    "source_name": _domain(row.get("url") or row.get("link") or ""),
                    "acquisition_source": "bocha",
                }
                for row in rows
                if isinstance(row, dict)
            ]
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="bocha",
                    capability="query_time_search",
                    status="used",
                    route=base_url,
                    configured=True,
                    metadata={"items": len(items), "task_id": task.task_id},
                )
            )
            return "bocha", items, []
        except Exception as exc:
            error = str(exc)
            logger.warning("[SourceGateway] Bocha search failed for {}: {}", task.query, error)
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="bocha",
                    capability="query_time_search",
                    status="error",
                    route=base_url,
                    configured=True,
                    errors=[error],
                    metadata={"task_id": task.task_id, "query": task.query},
                )
            )
            return "bocha", [], [error]

    def _search_tavily(self, task: RetrievalTask) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        api_key = getattr(self.settings, "TAVILY_API_KEY", None)
        base_url = "https://api.tavily.com/search"
        timeout = int(getattr(self.settings, "SEARCH_TIMEOUT", 30) or 30)
        self.diagnostics.append(
            ProviderDiagnostic(
                provider="tavily",
                capability="query_time_search",
                status="configured",
                route=base_url,
                configured=True,
                metadata={"task_id": task.task_id, "query": task.query},
            )
        )
        try:
            response = requests.post(
                base_url,
                json={
                    "api_key": api_key,
                    "query": task.query,
                    "max_results": task.max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results") or []
            items = [
                {
                    "url": row.get("url") or "",
                    "title": row.get("title") or "",
                    "text": row.get("content") or row.get("raw_content") or "",
                    "published_at": row.get("published_date"),
                    "source_type": "search_result",
                    "platform": _platform_key(row.get("url") or ""),
                    "source_name": _domain(row.get("url") or ""),
                    "acquisition_source": "tavily",
                }
                for row in rows
                if isinstance(row, dict)
            ]
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="tavily",
                    capability="query_time_search",
                    status="used",
                    route=base_url,
                    configured=True,
                    metadata={"items": len(items), "task_id": task.task_id},
                )
            )
            return "tavily", items, []
        except Exception as exc:
            error = str(exc)
            logger.warning("[SourceGateway] Tavily search failed for {}: {}", task.query, error)
            self.diagnostics.append(
                ProviderDiagnostic(
                    provider="tavily",
                    capability="query_time_search",
                    status="error",
                    route=base_url,
                    configured=True,
                    errors=[error],
                    metadata={"task_id": task.task_id, "query": task.query},
                )
            )
            return "tavily", [], [error]

    @staticmethod
    def _bocha_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            if isinstance(data.get("webPages"), dict):
                return data["webPages"].get("value") or []
            if isinstance(data.get("webpages"), list):
                return data["webpages"]
        if isinstance(payload.get("webPages"), dict):
            return payload["webPages"].get("value") or []
        return payload.get("results") or []

    def _to_normalized(self, raw: Dict[str, Any], task: RetrievalTask, index: int, provider: str = "unknown") -> NormalizedItem:
        url = str(raw.get("url") or "")
        title = str(raw.get("title") or "").strip()
        text = " ".join(str(raw.get("text") or raw.get("snippet") or title or url).split())
        raw_platform = str(raw.get("platform") or _domain(url) or "web")
        platform = _platform_key(url, raw_platform)
        source_type = str(raw.get("source_type") or self._source_type(url, platform))
        inferred_source_type = self._source_type(url, platform)
        if source_type == "search_result" and inferred_source_type in {"official", "mainstream_media", "ugc"}:
            source_type = inferred_source_type
        if source_type == "search_result" and canonical_social_platform(platform):
            source_type = "ugc"
        raw_id = _stable_id("raw", task.task_id, url, title, text[:120], str(index))
        item_id = _stable_id("item", raw_id)
        now = utc_now()
        acquisition_source = str(raw.get("acquisition_source") or provider or "unknown")
        source_name = str(raw.get("source_name") or raw_platform or _domain(url) or platform)
        return NormalizedItem(
            item_id=item_id,
            raw_id=raw_id,
            platform=platform or "web",
            source_type=source_type,
            source_name=source_name,
            source_item_id=raw_id,
            url=url,
            canonical_url=url.split("#")[0],
            author_id_hash=None,
            title=title,
            text=text[:4000],
            language=_language(f"{title} {text}"),
            published_at=raw.get("published_at"),
            observed_at=now,
            retrieved_at=now,
            retrieval_query=str(raw.get("retrieval_query") or task.query),
            raw_ref=f"{acquisition_source}://{task.task_id}/{raw_id}",
            acquisition_source=acquisition_source,
        )

    @staticmethod
    def _source_type(url: str, platform: str) -> str:
        host = _domain(url or platform)
        official_hosts = {"deepseek.com", "api-docs.deepseek.com", "platform.deepseek.com", "chat.deepseek.com"}
        if host in official_hosts or host.endswith(".deepseek.com"):
            return "official"
        if host.endswith(".gov") or ".gov." in host or "official" in host:
            return "official"
        if any(part in host for part in ["news", "reuters", "apnews", "bbc", "cnn", "media"]):
            return "mainstream_media"
        if canonical_social_platform(platform) or canonical_social_platform(host):
            return "ugc"
        return "search_result"

    def _allow_fixture_fallback(self) -> bool:
        if not self.settings:
            return False
        return bool(getattr(self.settings, "COORDINATOR_ALLOW_REPLAY_FALLBACK", False))

    def _fixture_items(self, task: Optional[RetrievalTask]) -> List[NormalizedItem]:
        query = task.query if task else "public opinion topic"
        target = query.split()[0] if query.split() else "the target"
        now = datetime.now(timezone.utc)
        rows = [
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/customer-support-coverage",
                "title": f"{target} support discussion draws media attention",
                "text": f"Independent coverage says the current {target} discussion centers on delayed support responses, unclear escalation paths, and whether the company has issued a complete explanation.",
                "published_at": (now - timedelta(hours=4)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/support-update",
                "title": f"{target} publishes a support operations update",
                "text": f"The company states that support volume increased during the incident window and says it is expanding response capacity while reviewing unresolved tickets.",
                "published_at": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/social-post-1",
                "title": "User complaint about slow response",
                "text": f"Users are sharing complaints that {target} support replies are slow and that ticket status is difficult to verify.",
                "published_at": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/forum-answer-2",
                "title": "Support process concern",
                "text": f"Users are sharing complaints that {target} support replies are slow and that ticket status is difficult to verify.",
                "published_at": (now - timedelta(minutes=55)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/social-post-3",
                "title": "Repeated complaint wording",
                "text": f"Users are sharing complaints that {target} support replies are slow and that ticket status is difficult to verify.",
                "published_at": (now - timedelta(minutes=52)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/discussion-thread-4",
                "title": "Counterpoint from users with resolved tickets",
                "text": f"Some users report that {target} support resolved their tickets after escalation, so the available sample is mixed rather than uniformly negative.",
                "published_at": (now - timedelta(minutes=35)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
            {
                "platform": "local_fixture",
                "source_type": "replay_fixture",
                "source_name": "Local replay fixture",
                "url": "replay://local/reputation-risk",
                "title": f"{target} faces reputation questions",
                "text": f"Analysts say the main risk is not a verified product defect, but the perception gap created by repeated unresolved support stories and limited direct evidence.",
                "published_at": (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
                "acquisition_source": "local_fixture",
            },
        ]
        items = []
        for index, row in enumerate(rows):
            fake_task = task or RetrievalTask(
                task_id="fixture",
                parent_claim_id=None,
                query=query,
                query_variants=[query],
                target_source="fixture",
                purpose="support",
                priority=1,
                deadline_sec=1,
                max_results=len(rows),
                budget={},
                created_by="fixture",
            )
            items.append(self._to_normalized(row, fake_task, index, provider="local_fixture"))
        return items
