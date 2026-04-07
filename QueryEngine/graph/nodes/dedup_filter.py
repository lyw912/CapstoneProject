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
from ...classifiers.stance_classifier import _is_official_domain, _extract_domain


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
    LangGraph 节点：去重过滤。

    优化策略：
    1. URL 精确去重。
    2. 内容去重（MinHash LSH）。
    3. 排序策略：官方域名优先，避免权威来源被非权威重复项挤掉。
    """
    sources: List[SourceItem] = state.get("raw_sources", [])

    # 排序：官方域名排在前面，确保去重时优先保留
    def _get_priority(s: SourceItem) -> int:
        domain = _extract_domain(s.get("url", ""))
        return 0 if _is_official_domain(domain) else 1

    sorted_sources = sorted(sources, key=_get_priority)

    # Stage 1: URL 精确去重
    seen_urls: set = set()
    url_deduped: List[SourceItem] = []
    for s in sorted_sources:
        norm = _normalize_url(s.get("url", ""))
        if norm and norm not in seen_urls:
            seen_urls.add(norm)
            url_deduped.append(s)

    # Phase 2：MinHash LSH 内容去重（80% 相似度阈值）
    content_deduped = minhash_dedup(url_deduped, threshold=0.80)

    # 恢复原始相对顺序（如果需要，或者保持排序后的顺序）
    # 这里我们保持排序后的顺序，因为后续的 Scorer 会重新评分

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
