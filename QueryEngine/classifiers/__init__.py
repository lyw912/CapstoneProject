"""
QueryEngine classifiers module

- TrustScorer: Multi-dimensional trust scoring (domain authority + timeliness + content quality + search ranking)
- HybridStanceClassifier: Hybrid stance classification (domain rules + keywords + sub-query weak labels)
"""

from .trust_scorer import compute_trust_score
from .stance_classifier import HybridStanceClassifier

__all__ = [
    "compute_trust_score",
    "HybridStanceClassifier",
]
