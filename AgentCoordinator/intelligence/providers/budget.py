"""Provider budget controls for semantic quality routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProviderBudget:
    """Small per-run budget used before optional semantic provider calls."""

    max_embedding_items: int = 120
    max_rerank_documents: int = 40
    timeout_seconds: int = 30
    embedding_calls: int = 0
    rerank_calls: int = 0

    @classmethod
    def from_settings(cls, settings: Any) -> "ProviderBudget":
        if settings is None:
            return cls()
        return cls(
            max_embedding_items=max(0, int(getattr(settings, "COORDINATOR_MAX_EMBEDDING_ITEMS", 120) or 0)),
            max_rerank_documents=max(0, int(getattr(settings, "COORDINATOR_MAX_RERANK_DOCUMENTS", 40) or 0)),
            timeout_seconds=max(1, int(getattr(settings, "COORDINATOR_PROVIDER_TIMEOUT", 30) or 30)),
        )

    def to_metadata(self) -> Dict[str, int]:
        return {
            "max_embedding_items": self.max_embedding_items,
            "max_rerank_documents": self.max_rerank_documents,
            "timeout_seconds": self.timeout_seconds,
            "embedding_calls": self.embedding_calls,
            "rerank_calls": self.rerank_calls,
        }
