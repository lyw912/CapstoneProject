"""
RRF (Reciprocal Rank Fusion) — SIGIR 2009

将多个子查询的有序结果列表融合为统一排名。
公式：RRF_score(d) = Σ_i  1 / (k + rank_i(d))

参考：Cormack G V, et al. Reciprocal rank fusion outperforms
      condorcet and individual rank learning methods. SIGIR 2009.
参考：架构文档 v2.0 Part 2 § 8.8
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Dict, List

# k=60 是 SIGIR 2009 论文的推荐超参数
_DEFAULT_K: int = 60


def rrf_fuse(
    source_groups: List[List[Dict]],
    k: int = _DEFAULT_K,
) -> List[Dict]:
    """
    RRF 多源结果融合。

    每组 source_groups[i] 是一个已排序的 SourceItem 列表（按搜索 API 返回顺序）。
    同一 URL 出现在多个列表中时，其 RRF 得分累加，体现跨源相关性。

    Args:
        source_groups: 多个子查询的搜索结果列表（每个列表内部已按相关性排序）
        k:             RRF 超参数，默认 60

    Returns:
        按 RRF 得分降序排列的去重结果列表（已将 rrf_score 字段更新）
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    url_to_source: Dict[str, Dict] = {}

    for ranked_list in source_groups:
        for rank, source in enumerate(ranked_list, start=1):
            url = (source.get("url") or "").strip()
            if not url:
                continue

            rrf_scores[url] += 1.0 / (k + rank)

            # 首次见到此 URL 时记录 source（后续同 URL 取第一个）
            if url not in url_to_source:
                url_to_source[url] = source

    # 将 RRF 得分写回 source，并排序
    result: List[Dict] = []
    for url, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        source = dict(url_to_source[url])          # shallow copy，不污染原始数据
        source["rrf_score"] = round(score, 6)
        result.append(source)

    return result


def group_sources_by_subquery(flat_sources: List[Dict]) -> List[List[Dict]]:
    """
    将平铺的 sources 列表按 sub_query_ref 分组，还原各子查询的有序列表。

    适用于已经混合在 raw_sources 中的情况，按插入顺序保留每组内的排名。

    Args:
        flat_sources: 平铺的 SourceItem 列表

    Returns:
        多个子列表，每个子列表对应一个 sub_query_ref 的搜索结果
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in flat_sources:
        key = s.get("sub_query_ref") or s.get("source_id") or str(uuid.uuid4())
        groups[key].append(s)
    return list(groups.values())
