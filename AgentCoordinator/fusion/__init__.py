"""Hierarchical Query/Media evidence-fusion runtime."""

from .supervisor import FusionCoordinator
from .evaluation import EVALUATION_VARIANTS, FusionEvaluationRecord, build_evaluation_record

__all__ = ["EVALUATION_VARIANTS", "FusionCoordinator", "FusionEvaluationRecord", "build_evaluation_record"]
