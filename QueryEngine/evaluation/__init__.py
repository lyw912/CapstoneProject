"""
QueryEngine evaluation 模块

评估指标：
  SCS  Stance Coverage Score    立场覆盖度（核心指标）
  SDI  Source Diversity Index   来源多样性（Shannon 熵归一化）
  SBS  Stance Balance Score     立场均衡度（1 - Gini系数）
  TSM  Trust Score Mean         平均可信度
  E2E  End-to-End Latency       端到端延迟（秒）

Phase 2 目标：SCS ≥ 0.70, SDI ≥ 0.50, E2E < 180s
"""

from .metrics import (
    stance_coverage_score,
    source_diversity_index,
    stance_balance_score,
    trust_score_mean,
    compute_all_metrics,
)

__all__ = [
    "stance_coverage_score",
    "source_diversity_index",
    "stance_balance_score",
    "trust_score_mean",
    "compute_all_metrics",
]
