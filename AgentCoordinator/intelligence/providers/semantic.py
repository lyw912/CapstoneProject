"""Optional Jina embedding and rerank routes for quality scoring.

Jina is the semantic provider for embeddings and reranking. Provider failures
are returned as diagnostics; deterministic quality features remain visible as
the fallback route.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from loguru import logger

from ..contracts import NormalizedItem, ProviderDiagnostic
from .budget import ProviderBudget


class SemanticQualityRouter:
    """API-backed semantic enrichment for clustering and relevance rerank."""

    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings
        self.budget = ProviderBudget.from_settings(settings)
        self.diagnostics: List[ProviderDiagnostic] = []

    def reset(self) -> None:
        self.diagnostics = []
        self.budget.embedding_calls = 0
        self.budget.rerank_calls = 0

    def embed_for_clustering(self, items: Sequence[NormalizedItem]) -> Dict[str, List[float]]:
        """Return Jina embeddings keyed by item id, or an empty dict."""
        if not self._jina_key():
            return {}
        if not items or self.budget.max_embedding_items <= 0:
            return {}

        selected = list(items)[: self.budget.max_embedding_items]
        if len(items) > len(selected):
            self._diagnose(
                provider="jina",
                capability="embedding",
                status="budget_limited",
                route=self._jina_embedding_url(),
                configured=True,
                warnings=[f"Embedding route limited to {len(selected)} of {len(items)} item(s)."],
                metadata=self.budget.to_metadata(),
            )

        payload = {
            "model": self._jina_embedding_model(),
            "input": [self._document_text(item) for item in selected],
            "task": "clustering",
            "embedding_type": "float",
            "normalized": True,
            "truncate": True,
        }
        dimensions = self._jina_embedding_dimensions()
        if dimensions:
            payload["dimensions"] = dimensions

        try:
            self.budget.embedding_calls += 1
            response = requests.post(
                self._jina_embedding_url(),
                headers=self._bearer_headers(self._jina_key()),
                json=payload,
                timeout=self.budget.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            vectors: Dict[str, List[float]] = {}
            for index, row in enumerate(payload.get("data") or []):
                if index >= len(selected) or not isinstance(row, dict):
                    continue
                vector = row.get("embedding")
                if isinstance(vector, list) and vector:
                    vectors[selected[index].item_id] = [float(value) for value in vector]
            self._diagnose(
                provider="jina",
                capability="embedding",
                status="used",
                route=self._jina_embedding_url(),
                configured=True,
                model=self._jina_embedding_model(),
                metadata={**self.budget.to_metadata(), "items": len(vectors)},
            )
            return vectors
        except Exception as exc:
            error = str(exc)
            logger.warning("[SemanticQualityRouter] Jina embedding failed: {}", error)
            self._diagnose(
                provider="jina",
                capability="embedding",
                status="error",
                route=self._jina_embedding_url(),
                configured=True,
                model=self._jina_embedding_model(),
                errors=[error],
                metadata=self.budget.to_metadata(),
            )
            return {}

    def rerank(self, query: str, items: Sequence[NormalizedItem]) -> Tuple[str, Dict[str, float]]:
        """Return provider name and relevance scores keyed by item id."""
        selected = self._rerank_candidates(items)
        if not selected:
            return "rules", {}
        if self._jina_key():
            provider, scores = self._jina_rerank(query, selected)
            if scores:
                return provider, scores
        return "rules", {}

    def semantic_duplicate(self, left: Optional[List[float]], right: Optional[List[float]]) -> bool:
        if not left or not right:
            return False
        threshold = float(getattr(self.settings, "COORDINATOR_SEMANTIC_DUPLICATE_THRESHOLD", 0.92) or 0.92)
        return _cosine(left, right) >= threshold

    def _rerank_candidates(self, items: Sequence[NormalizedItem]) -> List[NormalizedItem]:
        if self.budget.max_rerank_documents <= 0:
            return []
        ranked = sorted(
            items,
            key=lambda item: (
                1 if item.source_type in {"official", "mainstream_media"} else 0,
                len(f"{item.title} {item.text}"),
            ),
            reverse=True,
        )
        selected = ranked[: self.budget.max_rerank_documents]
        if len(items) > len(selected):
            self._diagnose(
                provider="semantic_router",
                capability="rerank_budget",
                status="budget_limited",
                route="quality_pipeline",
                configured=True,
                warnings=[f"Rerank route limited to {len(selected)} of {len(items)} item(s)."],
                metadata=self.budget.to_metadata(),
            )
        return selected

    def _jina_rerank(self, query: str, items: Sequence[NormalizedItem]) -> Tuple[str, Dict[str, float]]:
        payload = {
            "model": self._jina_rerank_model(),
            "query": query,
            "documents": [self._document_text(item) for item in items],
            "top_n": len(items),
            "return_documents": False,
        }
        try:
            self.budget.rerank_calls += 1
            response = requests.post(
                self._jina_rerank_url(),
                headers=self._bearer_headers(self._jina_key()),
                json=payload,
                timeout=self.budget.timeout_seconds,
            )
            response.raise_for_status()
            scores = self._scores_from_results(response.json().get("results") or [], items)
            self._diagnose(
                provider="jina",
                capability="rerank",
                status="used",
                route=self._jina_rerank_url(),
                configured=True,
                model=self._jina_rerank_model(),
                metadata={**self.budget.to_metadata(), "items": len(scores)},
            )
            return "jina", scores
        except Exception as exc:
            error = str(exc)
            logger.warning("[SemanticQualityRouter] Jina rerank failed: {}", error)
            self._diagnose(
                provider="jina",
                capability="rerank",
                status="error",
                route=self._jina_rerank_url(),
                configured=True,
                model=self._jina_rerank_model(),
                errors=[error],
                metadata=self.budget.to_metadata(),
            )
            return "jina", {}

    @staticmethod
    def _scores_from_results(results: List[Dict[str, Any]], items: Sequence[NormalizedItem]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(items):
                continue
            raw_score = row.get("relevance_score")
            try:
                score = max(0.0, min(1.0, float(raw_score)))
            except Exception:
                continue
            scores[items[index].item_id] = score
        return scores

    @staticmethod
    def _document_text(item: NormalizedItem) -> str:
        return " ".join(f"{item.title}\n{item.text}".split())[:3200]

    @staticmethod
    def _bearer_headers(api_key: Optional[str]) -> Dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _diagnose(
        self,
        *,
        provider: str,
        capability: str,
        status: str,
        route: str,
        configured: bool,
        model: Optional[str] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.diagnostics.append(
            ProviderDiagnostic(
                provider=provider,
                capability=capability,
                status=status,
                route=route,
                configured=configured,
                model=model,
                errors=errors or [],
                warnings=warnings or [],
                metadata=metadata or {},
            )
        )

    def _jina_key(self) -> Optional[str]:
        return getattr(self.settings, "JINA_API_KEY", None) if self.settings else None

    def _jina_embedding_url(self) -> str:
        return str(getattr(self.settings, "JINA_EMBEDDING_BASE_URL", None) or "https://api.jina.ai/v1/embeddings")

    def _jina_rerank_url(self) -> str:
        return str(getattr(self.settings, "JINA_RERANK_BASE_URL", None) or "https://api.jina.ai/v1/rerank")

    def _jina_embedding_model(self) -> str:
        return str(getattr(self.settings, "JINA_EMBEDDING_MODEL", None) or "jina-embeddings-v5-text-small")

    def _jina_rerank_model(self) -> str:
        return str(getattr(self.settings, "JINA_RERANK_MODEL", None) or "jina-reranker-v3")

    def _jina_embedding_dimensions(self) -> Optional[int]:
        value = getattr(self.settings, "JINA_EMBEDDING_DIMENSIONS", None) if self.settings else None
        if value in (None, "", 0):
            return None
        return int(value)


def _cosine(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)
