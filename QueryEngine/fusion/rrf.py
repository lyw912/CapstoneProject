"""
RRF (Reciprocal Rank Fusion) — SIGIR 2009

Merge ordered result lists from multiple sub-queries into a unified ranking.
Formula: RRF_score(d) = Σ_i  1 / (k + rank_i(d))

Reference: Cormack G V, et al. Reciprocal rank fusion outperforms
      condorcet and individual rank learning methods. SIGIR 2009.
Reference: Architecture Document v2.0 Part 2 § 8.8
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List

# k=60 is the recommended hyperparameter from the SIGIR 2009 paper
_DEFAULT_K: int = 60


def rrf_fuse(
    source_groups: List[List[Dict]],
    k: int = _DEFAULT_K,
) -> List[Dict]:
    """
    RRF multi-source result fusion.

    Each source_groups[i] is an ordered list of SourceItem (sorted by search API return order).
    When the same URL appears in multiple lists, its RRF scores are accumulated,
    reflecting cross-source relevance.

    Args:
        source_groups: Multiple sub-query search result lists (each list internally sorted by relevance)
        k:             RRF hyperparameter, default 60

    Returns:
        Deduplicated result list sorted by RRF score in descending order (rrf_score field updated)
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    url_to_source: Dict[str, Dict] = {}

    for ranked_list in source_groups:
        for rank, source in enumerate(ranked_list, start=1):
            url = (source.get("url") or "").strip()
            if not url:
                continue

            rrf_scores[url] += 1.0 / (k + rank)

            # First time seeing this URL, record source (for subsequent same URLs, take the first one)
            if url not in url_to_source:
                url_to_source[url] = source

    # Write RRF score back to source and sort
    result: List[Dict] = []
    for url, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        source = dict(url_to_source[url])          # shallow copy, avoid polluting original data
        source["rrf_score"] = round(score, 6)
        result.append(source)

    return result


def group_sources_by_subquery(flat_sources: List[Dict]) -> List[List[Dict]]:
    """
    Group flat sources list by sub_query_ref, restoring ordered lists for each sub-query.

    Suitable for cases where sources are already mixed in raw_sources,
    preserving ranking within each group by insertion order.

    Args:
        flat_sources: Flat list of SourceItem

    Returns:
        Multiple sublists, each corresponding to search results for a sub_query_ref
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in flat_sources:
        key = s.get("sub_query_ref") or s.get("source_id") or str(uuid.uuid4())
        groups[key].append(s)
    return list(groups.values())
