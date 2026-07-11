"""Evidence audit decisions replacing prompt-only debate."""

from __future__ import annotations

from typing import Dict, List

from ..contracts import AuditDecision, EvidenceGraph
from .sufficiency import EvidenceSufficiencyEvaluator


class EvidenceAuditor:
    """Produce structured audit decisions for every claim."""

    def __init__(self, evaluator: EvidenceSufficiencyEvaluator):
        self.evaluator = evaluator

    def run(self, graph: EvidenceGraph) -> EvidenceGraph:
        decisions: List[AuditDecision] = []
        task_ids_by_claim: Dict[str, List[str]] = {}
        for task in graph.retrieval_tasks:
            if task.parent_claim_id:
                task_ids_by_claim.setdefault(task.parent_claim_id, []).append(task.task_id)

        for index, claim in enumerate(graph.claims, start=1):
            sufficiency = self.evaluator.evaluate(claim)
            decision = self._decision(claim.status, sufficiency.reason_codes)
            decisions.append(
                AuditDecision(
                    decision_id=f"ad_{index:04d}",
                    claim_id=claim.claim_id,
                    auditor="judge",
                    decision=decision,
                    reason_codes=sufficiency.reason_codes,
                    explanation=self._explanation(decision, sufficiency.reason_codes),
                    required_edit=self._required_edit(decision, sufficiency.reason_codes),
                    follow_up_tasks=task_ids_by_claim.get(claim.claim_id, []),
                    confidence=self._confidence(decision, claim.confidence),
                )
            )

        graph.audit_decisions = decisions
        return graph

    @staticmethod
    def _decision(status: str, reason_codes: List[str]) -> str:
        if "missing_source_span" in reason_codes or status in {"unsupported", "demoted"}:
            return "reject"
        if status == "needs_search" and not reason_codes:
            return "needs_search"
        if any(code in reason_codes for code in ["high_copy_ratio", "ugc_only", "single_source", "one_sided"]):
            return "weaken"
        if status == "disputed":
            return "weaken"
        return "accept"

    @staticmethod
    def _explanation(decision: str, reason_codes: List[str]) -> str:
        if decision == "accept":
            return "The claim has source-span support and no blocking quality warning."
        if decision == "reject":
            return "The claim lacks sufficient source-span support or failed relevance checks."
        if decision == "needs_search":
            return "The claim needs more evidence before it can be used as a final assertion."
        reasons = ", ".join(reason_codes) if reason_codes else "limited evidence"
        return f"The claim is usable only with downgraded wording because of {reasons}."

    @staticmethod
    def _required_edit(decision: str, reason_codes: List[str]) -> str:
        if decision == "accept":
            return "Keep the claim citation-grounded and preserve sample boundaries."
        if decision == "reject":
            return "Remove the claim from final synthesis unless new source spans are found."
        if "high_copy_ratio" in reason_codes:
            return "Write repeated coverage strength, not independent viewpoint prevalence."
        if "ugc_only" in reason_codes or "single_source" in reason_codes:
            return "Write as observable discourse from the sampled sources, not a population-level fact."
        return "Use hedged wording and cite the supporting spans."

    @staticmethod
    def _confidence(decision: str, claim_confidence: float) -> float:
        delta = {"accept": 0.08, "weaken": 0.0, "needs_search": -0.08, "reject": -0.18}.get(decision, 0.0)
        return round(max(0.05, min(0.95, claim_confidence + delta)), 4)
