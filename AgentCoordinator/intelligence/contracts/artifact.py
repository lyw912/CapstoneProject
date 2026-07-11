"""Coordinator internal intelligence artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .evidence import (
    AuditDecision,
    EvidenceGraph,
    FreshnessSummary,
    Insight,
    ProviderDiagnostic,
    ResearchTraceStep,
    jsonable,
)


@dataclass
class CoordinatorIntelligenceArtifact:
    run_id: str
    query: str
    mode: str
    created_at: str
    target_entity: str
    analysis_type: str
    evidence_graph: EvidenceGraph
    evidence_graph_summary: Dict[str, Any]
    quality_summary: Dict[str, Any]
    freshness_summary: FreshnessSummary
    source_coverage: Dict[str, Any]
    source_coverage_limitations: List[str]
    provider_diagnostics: List[ProviderDiagnostic]
    research_trace: List[ResearchTraceStep]
    audit_summary: Dict[str, Any]
    insights: List[Insight]
    analysis_warnings: List[str]
    synthesis_markdown: str
    final_report_ready: bool
    report_engine_projection: Dict[str, Any]
    schema_version: str = "coordinator_intelligence_v1"
    budget_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)

    @property
    def audit_decisions(self) -> List[AuditDecision]:
        return self.evidence_graph.audit_decisions
