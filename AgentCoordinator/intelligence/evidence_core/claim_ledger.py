"""Merge specialist claim proposals into the canonical claim ledger."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List

from ..contracts import Claim, EvidenceGraph, EvidenceRelationEdge
from .blackboard import EvidenceBlackboardSnapshot


class ClaimLedgerMerger:
    """Evidence-bind proposals; unbound specialist prose never becomes a claim."""

    def merge(self, graph: EvidenceGraph, snapshot: EvidenceBlackboardSnapshot, target_entity: str) -> EvidenceGraph:
        proposed_span_by_id = {span.span_id: span for span in snapshot.evidence_spans}
        evidence_by_source = {item.item_id: item for item in graph.evidence_items}
        quality_by_item = graph.quality_index()
        by_key: Dict[str, Claim] = {self._claim_key(claim.claim_text): claim for claim in graph.claims}
        support_edges: List[EvidenceRelationEdge] = list(graph.support_edges)

        for proposal in snapshot.claim_proposals:
            bound_spans: List[str] = []
            source_ids: List[str] = []
            for proposed_span_id in proposal.evidence_span_ids:
                proposed_span = proposed_span_by_id.get(proposed_span_id)
                if not proposed_span:
                    continue
                source_ids.append(proposed_span.source_id)
                evidence = evidence_by_source.get(proposed_span.source_id)
                if evidence:
                    bound_spans.extend(span.span_id for span in evidence.spans)
            bound_spans = list(dict.fromkeys(bound_spans))
            if not bound_spans:
                continue

            key = self._claim_key(proposal.claim_text)
            existing = by_key.get(key)
            if existing:
                existing.supporting_spans = list(dict.fromkeys(existing.supporting_spans + bound_spans))
                existing.confidence = round(max(existing.confidence, proposal.confidence), 4)
                claim = existing
            else:
                qualities = [quality_by_item[source_id] for source_id in source_ids if source_id in quality_by_item]
                authority = max((item.source_authority_score for item in qualities), default=0.0)
                freshness = max((item.freshness_score for item in qualities), default=0.0)
                status = "supported" if authority >= 0.75 and proposal.confidence >= 0.65 else "needs_search"
                claim = Claim(
                    claim_id=self._claim_id(proposal.claim_text, proposal.task_id),
                    claim_text=proposal.claim_text,
                    claim_type=proposal.claim_type,
                    target_entity=proposal.target_entity or target_entity,
                    aspect=proposal.aspect,
                    time_scope="retrieval_time_window",
                    stance=proposal.stance,
                    sentiment="neutral",
                    supporting_spans=bound_spans,
                    contradicting_spans=[],
                    quality_summary={
                        "source_diversity": min(1.0, len(set(source_ids)) / 4.0),
                        "freshness_score": freshness,
                        "source_authority": authority,
                        "copy_ratio": 0.0,
                        "proposal_uncertainty": list(proposal.uncertainty),
                    },
                    status=status,
                    confidence=round(min(0.9, proposal.confidence), 4),
                    created_by=f"{proposal.agent}:claim_proposal",
                    model="specialist-evidence-bound",
                )
                graph.claims.append(claim)
                by_key[key] = claim

            support_edges.append(
                EvidenceRelationEdge(
                    edge_id=self._edge_id(proposal.proposal_id, claim.claim_id),
                    relation="proposes",
                    from_id=proposal.proposal_id,
                    to_id=claim.claim_id,
                    evidence_span_ids=bound_spans,
                    confidence=proposal.confidence,
                    explanation="Specialist proposal accepted into the ledger only after source-span binding.",
                )
            )

        graph.support_edges = support_edges
        return graph

    @staticmethod
    def _claim_key(text: str) -> str:
        return re.sub(r"\W+", "", str(text or "").lower())[:240]

    @staticmethod
    def _claim_id(text: str, task_id: str) -> str:
        digest = hashlib.sha1(f"{task_id}\x1f{text}".encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"clp_{digest}"

    @staticmethod
    def _edge_id(proposal_id: str, claim_id: str) -> str:
        digest = hashlib.sha1(f"{proposal_id}\x1f{claim_id}".encode("utf-8")).hexdigest()[:12]
        return f"sup_{digest}"
