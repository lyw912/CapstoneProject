"""Claim mining from representative evidence and source spans."""

from __future__ import annotations

from typing import Dict, List

from ..contracts import Claim, ContradictionEdge, EvidenceGraph, QualityFeatures


class ClaimMiner:
    """Create structured claims from evidence representatives."""

    def run(self, graph: EvidenceGraph, target_entity: str) -> EvidenceGraph:
        quality_by_item = graph.quality_index()
        claims: List[Claim] = []

        for index, evidence in enumerate(graph.evidence_items, start=1):
            quality = quality_by_item.get(evidence.item_id)
            if not quality or not evidence.spans:
                continue
            span_ids = [span.span_id for span in evidence.spans]
            claim_type = self._claim_type(quality, evidence.source_type)
            status = self._status(quality, evidence.source_type)
            confidence = self._confidence(quality, status)
            text = self._claim_text(target_entity, quality, evidence)
            claims.append(
                Claim(
                    claim_id=f"cl_{index:04d}",
                    claim_text=text,
                    claim_type=claim_type,
                    target_entity=target_entity,
                    aspect=quality.aspect,
                    time_scope=self._time_scope(evidence.published_at),
                    stance=quality.stance,
                    sentiment=quality.sentiment,
                    supporting_spans=span_ids,
                    contradicting_spans=[],
                    quality_summary={
                        "source_diversity": self._source_diversity(graph, quality),
                        "freshness_score": quality.freshness_score,
                        "amplification_adjusted_support": round(
                            quality.persuasiveness_score * max(0.35, 1.0 - quality.copy_ratio_in_cluster * 0.5),
                            4,
                        ),
                        "copy_ratio": quality.copy_ratio_in_cluster,
                        "source_authority": quality.source_authority_score,
                    },
                    status=status,
                    confidence=confidence,
                    created_by="claim_miner_v1",
                    model="rules-representative-spans",
                )
            )

        self._attach_contradictions(claims)
        graph.claims = claims
        graph.contradiction_edges = self._contradiction_edges(claims)
        return graph

    @staticmethod
    def _claim_type(quality: QualityFeatures, source_type: str) -> str:
        if source_type == "official":
            return "fact"
        if quality.aspect == "reputation_risk":
            return "risk"
        if quality.stance in {"support", "oppose", "mixed"}:
            return "stance"
        return "sentiment"

    @staticmethod
    def _status(quality: QualityFeatures, source_type: str) -> str:
        if quality.relevance_score < 0.30:
            return "demoted"
        if source_type == "official" and quality.source_authority_score >= 0.80:
            return "supported"
        if quality.copy_ratio_in_cluster >= 0.55 and quality.source_authority_score < 0.60:
            return "single_source"
        if quality.persuasiveness_score >= 0.58:
            return "supported"
        if quality.persuasiveness_score >= 0.42:
            return "needs_search"
        return "unsupported"

    @staticmethod
    def _confidence(quality: QualityFeatures, status: str) -> float:
        base = quality.persuasiveness_score
        if status == "supported":
            base += 0.08
        elif status in {"single_source", "needs_search"}:
            base -= 0.06
        elif status in {"unsupported", "demoted"}:
            base -= 0.18
        return round(max(0.05, min(0.95, base)), 4)

    @staticmethod
    def _source_diversity(graph: EvidenceGraph, quality: QualityFeatures) -> float:
        cluster = next((item for item in graph.canonical_clusters if item.canonical_item_id == quality.canonical_item_id), None)
        if not cluster:
            return 0.0
        return round(min(1.0, len(cluster.platforms) / 4.0), 4)

    @staticmethod
    def _claim_text(target_entity: str, quality: QualityFeatures, evidence) -> str:
        aspect = quality.aspect.replace("_", " ")
        platform = evidence.platform or "observable sources"
        if quality.stance == "official":
            return f"{target_entity} has an official or high-authority source addressing {aspect}."
        if quality.stance == "oppose":
            return f"Observable {platform} discourse raises negative {aspect} claims about {target_entity}."
        if quality.stance == "support":
            return f"Observable {platform} discourse contains supportive or resolving {aspect} signals about {target_entity}."
        return f"Observable sources discuss {target_entity} in relation to {aspect}."

    @staticmethod
    def _time_scope(published_at: str | None) -> str:
        if not published_at:
            return "retrieval_time_window"
        return f"{published_at}/retrieval_time"

    @staticmethod
    def _attach_contradictions(claims: List[Claim]) -> None:
        by_aspect: Dict[str, List[Claim]] = {}
        for claim in claims:
            by_aspect.setdefault(claim.aspect, []).append(claim)
        for group in by_aspect.values():
            support_spans = [span for claim in group if claim.stance in {"support", "official"} for span in claim.supporting_spans]
            oppose_spans = [span for claim in group if claim.stance == "oppose" for span in claim.supporting_spans]
            if not support_spans or not oppose_spans:
                continue
            for claim in group:
                if claim.stance == "oppose":
                    claim.contradicting_spans = support_spans[:3]
                    if claim.status == "supported":
                        claim.status = "disputed"
                elif claim.stance in {"support", "official"}:
                    claim.contradicting_spans = oppose_spans[:3]
                    if claim.status == "supported" and claim.stance != "official":
                        claim.status = "disputed"

    @staticmethod
    def _contradiction_edges(claims: List[Claim]) -> List[ContradictionEdge]:
        edges: List[ContradictionEdge] = []
        for left in claims:
            if not left.contradicting_spans:
                continue
            for right in claims:
                if left.claim_id == right.claim_id or left.aspect != right.aspect:
                    continue
                if set(right.supporting_spans) & set(left.contradicting_spans):
                    edges.append(
                        ContradictionEdge(
                            edge_id=f"ct_{len(edges) + 1:04d}",
                            claim_a=left.claim_id,
                            claim_b=right.claim_id,
                            relation="source_disagreement",
                            explanation=f"{left.aspect.replace(chr(95), chr(32)).title()} evidence contains opposing source spans.",
                            severity="medium",
                            requires_follow_up=True,
                        )
                    )
                    break
        return edges
