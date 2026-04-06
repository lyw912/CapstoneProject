"""
StanceClassify 节点

对 scored_sources 中每条来源调用 HybridStanceClassifier，
填充 stance_label 与 stance_confidence，产出 classified_sources。

关键设计：在分类前，从 state["sub_queries"] 构建
  sub_query_ref → target_stance 映射，注入到每条 source 的 _target_stance 字段，
  为分类器提供"子查询弱标签"（置信度 0.50）。

Phase 2 新增节点，位于图中 trust_scorer → stance_classify → coverage_check。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from loguru import logger

from ...classifiers.stance_classifier import HybridStanceClassifier
from ..state import QueryAgentState


def _build_stance_lookup(state: QueryAgentState) -> Dict[str, str]:
    """
    从 sub_queries（含所有历史轮次）和 gap_queries 构建
      query文本 → target_stance 映射。
    """
    lookup: Dict[str, str] = {}

    for sq in state.get("sub_queries") or []:
        q = sq.get("query", "")
        stance = sq.get("target_stance", "")
        if q and stance:
            lookup[q] = stance

    # gap_queries 是补搜时由 GapFiller 生成的，同样需要纳入映射
    for gq in state.get("gap_queries") or []:
        q = gq.get("query", "")
        stance = gq.get("target_stance", "")
        if q and stance:
            lookup[q] = stance

    return lookup


async def stance_classify_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：混合立场分类。

    输入：state["scored_sources"]（或降级到 deduped_sources）
    输出：state["classified_sources"]（每条 stance_label + stance_confidence 已填充）
    """
    sources: List[dict] = (
        state.get("scored_sources")
        or state.get("deduped_sources")
        or []
    )
    query = state.get("original_query", "")

    # 构建子查询 → 目标立场 的查找表（用于弱标签）
    stance_lookup = _build_stance_lookup(state)

    classifier = HybridStanceClassifier()
    classified: List[dict] = []

    for s in sources:
        s = dict(s)  # shallow copy

        # 注入子查询弱标签
        sub_ref = s.get("sub_query_ref", "")
        s["_target_stance"] = stance_lookup.get(sub_ref, "")

        # 分类
        stance, confidence = classifier.classify(s, query)
        s["stance_label"] = stance
        s["stance_confidence"] = confidence

        classified.append(s)

    # 统计立场分布（用于日志）
    stance_counts = Counter(s["stance_label"] for s in classified)
    trace = (
        f"[StanceClassify] 分类 {len(classified)} 条来源, "
        f"分布={dict(stance_counts)}"
    )
    logger.info(trace)

    return {
        "classified_sources": classified,
        "trace_log": [trace],
    }
