"""
QueryEngine classifiers 模块

- TrustScorer: 多维可信度评分（域名权威性 + 时效性 + 内容质量 + 搜索排名）
- HybridStanceClassifier: 混合立场分类（域名规则 + 关键词 + 子查询弱标签）
"""

from .trust_scorer import compute_trust_score
from .stance_classifier import HybridStanceClassifier

__all__ = [
    "compute_trust_score",
    "HybridStanceClassifier",
]
