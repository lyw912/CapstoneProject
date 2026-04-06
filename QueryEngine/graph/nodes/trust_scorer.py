"""
TrustScorer 节点

对 deduped_sources 中每条来源调用 compute_trust_score()，
将计算结果写入 source["trust_score"]，产出 scored_sources。

Phase 2 新增节点，位于图中 dedup_filter → trust_scorer → stance_classify。
"""

from __future__ import annotations

from typing import List

from loguru import logger

from ...classifiers.trust_scorer import compute_trust_score
from ..state import QueryAgentState


async def trust_scorer_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：TrustScore 计算。

    输入：state["deduped_sources"]
    输出：state["scored_sources"]（每条 trust_score 已填充）
    """
    sources: List[dict] = state.get("deduped_sources") or []

    scored: List[dict] = []
    for s in sources:
        s = dict(s)  # shallow copy，避免修改 State 中的原始对象
        s["trust_score"] = compute_trust_score(s)
        scored.append(s)

    if scored:
        avg = sum(s["trust_score"] for s in scored) / len(scored)
        max_s = max(s["trust_score"] for s in scored)
        min_s = min(s["trust_score"] for s in scored)
    else:
        avg = max_s = min_s = 0.0

    trace = (
        f"[TrustScorer] 处理 {len(scored)} 条来源, "
        f"均值={avg:.3f}, 最高={max_s:.3f}, 最低={min_s:.3f}"
    )
    logger.info(trace)

    return {
        "scored_sources": scored,
        "trace_log": [trace],
    }
