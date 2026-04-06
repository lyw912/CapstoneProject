"""
CoverageCheck 节点

检查 classified_sources 中各立场的覆盖情况，
识别未达到最低要求的立场，写入 missing_stances。

CoverageCheck 本身是同步节点（无 LLM 调用），是条件路由的决策依据。

覆盖度计算公式（SCS，Stance Coverage Score）：
  SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
  K = 参与评估的立场数量（4 个：support, oppose, official, neutral）

Phase 2 新增节点，位于图中 stance_classify → coverage_check → [router]。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from loguru import logger

from ..state import QueryAgentState

# ---------------------------------------------------------------------------
# 各立场最低要求来源数（"background" 不列入闭环检查，属于补充信息）
# ---------------------------------------------------------------------------

MINIMUM_STANCE_COUNTS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


def _compute_coverage_score(stance_counts: Dict[str, int]) -> float:
    """
    基于 MINIMUM_STANCE_COUNTS 计算立场覆盖度分数（0–1）。

    SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
    """
    total_required = sum(MINIMUM_STANCE_COUNTS.values())
    total_met = sum(
        min(stance_counts.get(s, 0), c)
        for s, c in MINIMUM_STANCE_COUNTS.items()
    )
    return round(total_met / max(total_required, 1), 3)


def coverage_check_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：立场覆盖度检查。

    输入：state["classified_sources"]
    输出：state["stance_coverage"]（各立场实际数量）
          state["missing_stances"]（未达到最低阈值的立场列表）
    """
    sources: List[dict] = state.get("classified_sources") or []

    # 只统计有明确立场标签的来源（排除 None / "unclassified"）
    stance_counts = Counter(
        s.get("stance_label")
        for s in sources
        if s.get("stance_label") and s.get("stance_label") not in ("unclassified",)
    )

    # 识别未达到最低阈值的立场
    missing: List[str] = [
        stance
        for stance, min_count in MINIMUM_STANCE_COUNTS.items()
        if stance_counts.get(stance, 0) < min_count
    ]

    coverage_score = _compute_coverage_score(dict(stance_counts))

    trace = (
        f"[CoverageCheck] 立场分布={dict(stance_counts)}, "
        f"缺失={missing}, SCS={coverage_score:.2f}"
    )
    logger.info(trace)

    return {
        "stance_coverage":  dict(stance_counts),
        "missing_stances":  missing,
        "trace_log": [trace],
    }
