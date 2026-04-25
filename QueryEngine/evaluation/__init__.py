"""
QueryEngine evaluation module

Evaluation Metrics:
  SCS  Stance Coverage Score    Stance coverage (core metric)
  SDI  Source Diversity Index   Source diversity (Shannon entropy normalized)
  SBS  Stance Balance Score     Stance balance (1 - Gini coefficient)
  TSM  Trust Score Mean         Average trust score
  E2E  End-to-End Latency       End-to-end latency (seconds)

Phase 2 Targets: SCS ≥ 0.70, SDI ≥ 0.50, E2E < 180s
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
