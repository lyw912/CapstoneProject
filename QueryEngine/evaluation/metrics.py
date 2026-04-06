"""
评估指标实现

参考：架构文档 v2.0 Part 3 § 12

指标体系：
  SCS  Stance Coverage Score    — 核心，衡量各立场是否被充分覆盖
  SDI  Source Diversity Index   — Shannon 熵归一化，衡量来源平台分散度
  SBS  Stance Balance Score     — 1 - Gini系数，衡量立场分布是否均衡
  TSM  Trust Score Mean         — 平均可信度
  E2E  End-to-End Latency       — 端到端延迟（调用方传入）
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional

# Phase 2 目标阈值
PHASE2_TARGETS = {
    "scs": 0.70,
    "sdi": 0.50,
    "sbs": 0.50,
    "tsm": 0.50,
    "e2e": 180.0,
}

# 立场覆盖度计算用的最低阈值（与 coverage_check.py 保持一致）
_STANCE_THRESHOLDS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


# ---------------------------------------------------------------------------
# SCS — Stance Coverage Score
# ---------------------------------------------------------------------------

def stance_coverage_score(output: dict) -> float:
    """
    立场覆盖度评分（0–1）。

    SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)

    目标：≥ 0.70（Phase 2），≥ 0.75（Phase 3）
    """
    sources: List[dict] = output.get("sources") or []
    stance_counts = Counter(
        s.get("stance_label")
        for s in sources
        if s.get("stance_label") and s["stance_label"] not in ("unclassified",)
    )

    if not stance_counts:
        return 0.0

    scores = []
    for stance, threshold in _STANCE_THRESHOLDS.items():
        actual = stance_counts.get(stance, 0)
        scores.append(min(actual / threshold, 1.0))

    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# SDI — Source Diversity Index
# ---------------------------------------------------------------------------

def source_diversity_index(output: dict) -> float:
    """
    来源多样性指数（0–1），基于归一化 Shannon 熵。

    SDI = H(platforms) / log2(|unique_platforms|)

    目标：≥ 0.50（Phase 2）
    """
    sources: List[dict] = output.get("sources") or []
    if not sources:
        return 0.0

    platform_counts = Counter(s.get("platform", "") or "unknown" for s in sources)
    n_platforms = len(platform_counts)

    if n_platforms <= 1:
        return 0.0

    total = sum(platform_counts.values())
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in platform_counts.values()
        if c > 0
    )
    max_entropy = math.log2(n_platforms)
    return round(entropy / max_entropy if max_entropy > 0 else 0.0, 3)


# ---------------------------------------------------------------------------
# SBS — Stance Balance Score
# ---------------------------------------------------------------------------

def stance_balance_score(output: dict) -> float:
    """
    立场均衡度（0–1），基于 1 - Gini 系数。

    完全均衡 → SBS = 1.0；完全偏向一个立场 → SBS → 0。

    目标：≥ 0.50（Phase 2）
    """
    dist: Dict[str, float] = output.get("stance_distribution") or {}
    # 过滤掉 unclassified
    values = [v for k, v in dist.items() if k != "unclassified"]

    if not values or len(values) <= 1:
        return 0.0

    n = len(values)
    mean_v = sum(values) / n
    if mean_v == 0:
        return 0.0

    # Gini = Σ|v_i - v_j| / (2 * n^2 * mean)
    gini = sum(abs(v1 - v2) for v1 in values for v2 in values) / (2 * n * n * mean_v)
    return round(max(0.0, 1.0 - gini), 3)


# ---------------------------------------------------------------------------
# TSM — Trust Score Mean
# ---------------------------------------------------------------------------

def trust_score_mean(output: dict) -> float:
    """
    平均可信度（0–1）。

    目标：≥ 0.50（Phase 2）
    """
    sources: List[dict] = output.get("sources") or []
    if not sources:
        return 0.0

    scores = [float(s.get("trust_score") or 0.0) for s in sources]
    return round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# 综合计算
# ---------------------------------------------------------------------------

def compute_all_metrics(
    output: dict,
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """
    一次性计算所有评估指标。

    Args:
        output:          QueryAgentOutput 字典
        elapsed_seconds: 端到端耗时（秒），None 表示未测量

    Returns:
        包含 scs/sdi/sbs/tsm/e2e 的字典
    """
    return {
        "scs": stance_coverage_score(output),
        "sdi": source_diversity_index(output),
        "sbs": stance_balance_score(output),
        "tsm": trust_score_mean(output),
        "e2e": round(elapsed_seconds, 1) if elapsed_seconds is not None else -1.0,
        "sources_count":    output.get("total_sources_kept", 0),
        "search_iterations": output.get("search_iterations", 0),
        "coverage_score":   output.get("coverage_score", 0.0),
    }


def check_phase2_pass(metrics: Dict[str, float]) -> bool:
    """
    检查是否通过 Phase 2 验收标准：
      SCS ≥ 0.70, SDI ≥ 0.50, E2E < 180s
    """
    e2e = metrics.get("e2e", -1.0)
    return (
        metrics.get("scs", 0) >= PHASE2_TARGETS["scs"]
        and metrics.get("sdi", 0) >= PHASE2_TARGETS["sdi"]
        and (e2e < 0 or e2e < PHASE2_TARGETS["e2e"])
    )


def format_metrics_report(query: str, metrics: Dict[str, float]) -> str:
    """格式化输出单条查询的评估结果。"""
    passed = check_phase2_pass(metrics)
    status = "✅ PASS" if passed else "❌ FAIL"

    lines = [
        f"查询: {query}",
        f"  SCS={metrics.get('scs', 0):.3f}  (目标≥0.70)",
        f"  SDI={metrics.get('sdi', 0):.3f}  (目标≥0.50)",
        f"  SBS={metrics.get('sbs', 0):.3f}  (目标≥0.50)",
        f"  TSM={metrics.get('tsm', 0):.3f}  (目标≥0.50)",
        f"  E2E={metrics.get('e2e', -1):.1f}s (目标<180s)",
        f"  来源数={metrics.get('sources_count', 0)}, "
        f"搜索轮次={metrics.get('search_iterations', 0)}",
        f"  结论: {status}",
    ]
    return "\n".join(lines)
