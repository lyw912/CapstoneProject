"""Evidence sufficiency checks for claim-driven research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..contracts import Claim


@dataclass
class SufficiencyDecision:
    claim_id: str
    relevance_ok: bool
    support_span_count: int
    source_diversity_score: float
    freshness_score: float
    contradiction_status: str
    amplification_bias: str
    sufficiency: str
    recommended_action: str
    reason_codes: List[str] = field(default_factory=list)


class EvidenceSufficiencyEvaluator:
    """Translate claim quality metadata into routing decisions."""

    def evaluate(self, claim: Claim) -> SufficiencyDecision:
        support_span_count = len(claim.supporting_spans)
        source_diversity = float((claim.quality_summary or {}).get("source_diversity") or 0.0)
        freshness = float((claim.quality_summary or {}).get("freshness_score") or 0.0)
        copy_ratio = float((claim.quality_summary or {}).get("copy_ratio") or 0.0)
        relevance_ok = claim.status not in {"demoted", "unsupported"}
        contradiction_status = "has_counter_evidence" if claim.contradicting_spans else "none"
        amplification_bias = "high" if copy_ratio >= 0.5 else "moderate" if copy_ratio >= 0.25 else "low"
        reason_codes: List[str] = []
        if support_span_count < 1:
            reason_codes.append("missing_source_span")
        is_official = claim.stance == "official" or claim.claim_type == "fact"
        if not is_official and source_diversity < 0.35:
            reason_codes.append("single_source")
        if claim.stance not in {"official", "neutral"} and source_diversity < 0.5:
            reason_codes.append("ugc_only")
        if freshness < 0.35:
            reason_codes.append("stale_evidence")
        if amplification_bias == "high" and not is_official:
            reason_codes.append("high_copy_ratio")
        if contradiction_status == "none" and claim.claim_type in {"risk", "causal", "stance"}:
            reason_codes.append("one_sided")
        if claim.status == "needs_search":
            reason_codes.append("needs_more_evidence")

        if not relevance_ok or "missing_source_span" in reason_codes:
            sufficiency = "weak"
            action = "reject_or_search"
        elif claim.status == "supported" and not reason_codes:
            sufficiency = "high"
            action = "accept"
        elif claim.status == "disputed" or "high_copy_ratio" in reason_codes or "ugc_only" in reason_codes:
            sufficiency = "moderate"
            action = "weaken_claim"
        else:
            sufficiency = "moderate"
            action = "search_or_weaken"

        return SufficiencyDecision(
            claim_id=claim.claim_id,
            relevance_ok=relevance_ok,
            support_span_count=support_span_count,
            source_diversity_score=round(source_diversity, 4),
            freshness_score=round(freshness, 4),
            contradiction_status=contradiction_status,
            amplification_bias=amplification_bias,
            sufficiency=sufficiency,
            recommended_action=action,
            reason_codes=sorted(set(reason_codes)),
        )
