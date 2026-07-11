"""Evaluation contracts for later server-side fusion experiments.

This module records observable measurements only. It does not assign grades,
fabricate baseline results, or write files unless an external harness chooses to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from AgentCoordinator.intelligence.contracts import CoordinatorIntelligenceArtifact, jsonable


EVALUATION_VARIANTS = {
    "fused",
    "query_only",
    "media_only",
    "previous_intelligence",
}


@dataclass
class FusionEvaluationRecord:
    run_id: str
    topic_id: str
    variant: str
    status: str
    duration_seconds: Optional[float]
    canonical_sources: int
    acquisition_observations: int
    claims: int
    insights: int
    cited_insight_ratio: float
    query_coverage_score: Optional[float]
    media_dossier_coverage_score: Optional[float]
    provider_failures: int
    contradiction_edges: int
    support_edges: int
    human_scores: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "factual_correctness": None,
            "citation_validity": None,
            "coverage": None,
            "counter_evidence_quality": None,
            "narrative_usefulness": None,
        }
    )
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


def build_evaluation_record(
    artifact: CoordinatorIntelligenceArtifact,
    *,
    topic_id: str,
    variant: str = "fused",
    duration_seconds: Optional[float] = None,
) -> FusionEvaluationRecord:
    if variant not in EVALUATION_VARIANTS:
        raise ValueError(f"Unsupported evaluation variant: {variant}")
    graph = artifact.evidence_graph
    query_coverage = [item.score for item in graph.coverage_assessments if item.agent == "query_agent"]
    media_coverage = [item.score for item in graph.coverage_assessments if item.agent == "media_agent"]
    cited = sum(1 for item in graph.insights if item.citation_spans)
    return FusionEvaluationRecord(
        run_id=artifact.run_id,
        topic_id=topic_id,
        variant=variant,
        status="measured_not_scored",
        duration_seconds=duration_seconds,
        canonical_sources=len(graph.canonical_clusters),
        acquisition_observations=len(graph.acquisition_observations),
        claims=len(graph.claims),
        insights=len(graph.insights),
        cited_insight_ratio=round(cited / len(graph.insights), 4) if graph.insights else 0.0,
        query_coverage_score=query_coverage[-1] if query_coverage else None,
        media_dossier_coverage_score=media_coverage[-1] if media_coverage else None,
        provider_failures=sum(1 for item in artifact.provider_diagnostics if item.status == "error"),
        contradiction_edges=len(graph.contradiction_edges),
        support_edges=len(graph.support_edges),
        notes=["Human scores and comparative conclusions are intentionally unset until server experiments run."],
    )
