"""EvidenceCore and audit kernels shared by the fusion supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..contracts import EvidenceGraph, FreshnessSummary, ProviderDiagnostic
from ..quality.pipeline import QualityPipeline
from ..reasoning.audit import EvidenceAuditor
from ..reasoning.claim_miner import ClaimMiner
from ..reasoning.sufficiency import EvidenceSufficiencyEvaluator
from ..reasoning.synthesis import CitationGroundedSynthesis
from .blackboard import EvidenceBlackboard
from .claim_ledger import ClaimLedgerMerger


@dataclass
class EvidenceCoreResult:
    graph: EvidenceGraph
    quality_summary: Dict[str, object]
    freshness_summary: FreshnessSummary
    provider_diagnostics: List[ProviderDiagnostic]


class EvidenceCorePipeline:
    """Normalize, quality-score, deduplicate and merge a blackboard snapshot."""

    def __init__(self, settings=None):
        self.quality = QualityPipeline(settings=settings)
        self.claim_miner = ClaimMiner()
        self.claim_merger = ClaimLedgerMerger()

    def run(self, blackboard: EvidenceBlackboard, query: str, target_entity: str) -> EvidenceCoreResult:
        quality_result = self.quality.run(
            blackboard.normalized_items(),
            query=query,
            target_entity=target_entity,
        )
        graph = self.claim_miner.run(quality_result.graph, target_entity=target_entity)
        snapshot = blackboard.snapshot()
        graph.acquisition_observations = snapshot.acquisitions
        graph.proposed_evidence_spans = snapshot.evidence_spans
        graph.claim_proposals = snapshot.claim_proposals
        graph.coverage_assessments = snapshot.coverage_assessments
        graph.section_dossiers = snapshot.section_dossiers
        graph.research_tasks = snapshot.research_tasks
        graph.agent_runs = snapshot.agent_runs
        graph.blackboard_version = snapshot.version
        graph = self.claim_merger.merge(graph, snapshot, target_entity=target_entity)
        self.claim_miner._attach_contradictions(graph.claims)
        graph.contradiction_edges = self.claim_miner._contradiction_edges(graph.claims)
        return EvidenceCoreResult(
            graph=graph,
            quality_summary=quality_result.quality_summary,
            freshness_summary=quality_result.freshness_summary,
            provider_diagnostics=quality_result.provider_diagnostics,
        )


class AuditKernel:
    """Claim-level sufficiency, audit decisions and citation-bound synthesis."""

    def __init__(self):
        self.sufficiency = EvidenceSufficiencyEvaluator()
        self.auditor = EvidenceAuditor(self.sufficiency)
        self.synthesizer = CitationGroundedSynthesis()

    def finalize(self, graph: EvidenceGraph, freshness: FreshnessSummary) -> Tuple[EvidenceGraph, str]:
        graph = self.auditor.run(graph)
        return self.synthesizer.run(graph, freshness)
