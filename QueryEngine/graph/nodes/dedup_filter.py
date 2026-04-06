"""
DedupFilter 节点 — 去重过滤

Phase 1：URL 精确去重（标准化后比较）。
Phase 2：MinHash LSH 内容去重（datasketch，阈值 0.80）。
"""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse, urlunparse

from loguru import logger

from ..state import QueryAgentState, SourceItem
from ...fusion.dedup import minhash_dedup


# ---------------------------------------------------------------------------
# URL 标准化
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """
    URL 标准化：去除 www.、查询参数、锚点、末尾斜杠，统一小写。
    用于精确去重。
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
# 节点函数
# ---------------------------------------------------------------------------

def dedup_filter_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：URL 精确去重。

    从 raw_sources（含所有历史轮次的累积结果）中去重，
    产出 deduped_sources 供后续节点使用。
    """
    sources: List[SourceItem] = state.get("raw_sources", [])

    # Stage 1: URL 精确去重
    seen_urls: set = set()
    url_deduped: List[SourceItem] = []
    for s in sources:
        norm = _normalize_url(s.get("url", ""))
        if norm and norm not in seen_urls:
            seen_urls.add(norm)
            url_deduped.append(s)

    # Phase 2：MinHash LSH 内容去重（80% 相似度阈值）
    content_deduped = minhash_dedup(url_deduped, threshold=0.80)

    trace = (
        f"[DedupFilter] 输入{len(sources)}条, "
        f"URL去重后{len(url_deduped)}条, "
        f"内容去重后{len(content_deduped)}条"
    )
    logger.info(trace)

    return {
        "deduped_sources": content_deduped,
        "trace_log": [trace],
    }
