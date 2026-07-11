"""Citation-grounded synthesis from audited claims."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..contracts import EvidenceGraph, Insight


class CitationGroundedSynthesis:
    """Generate final insights only from audited, cited claims."""

    def run(self, graph: EvidenceGraph, freshness_summary) -> Tuple[EvidenceGraph, str]:
        decisions = {decision.claim_id: decision for decision in graph.audit_decisions}
        claim_by_id = graph.claim_index()
        accepted_or_weakened = [
            claim
            for claim in graph.claims
            if decisions.get(claim.claim_id)
            and decisions[claim.claim_id].decision in {"accept", "weaken"}
            and claim.supporting_spans
        ]

        grouped: Dict[str, List] = defaultdict(list)
        for claim in accepted_or_weakened:
            grouped[claim.aspect].append(claim)

        ordered_groups = sorted(grouped.items(), key=lambda item: self._aspect_sort_key(item[0], item[1], graph))

        insights: List[Insight] = []
        for index, (aspect, claims) in enumerate(ordered_groups, start=1):
            citation_spans = []
            counter_spans = []
            warnings = []
            strength = "moderate"
            wording_policy = "hedge"
            for claim in claims:
                citation_spans.extend(claim.supporting_spans)
                counter_spans.extend(claim.contradicting_spans)
                decision = decisions.get(claim.claim_id)
                if decision:
                    warnings.extend(decision.reason_codes)
                    if decision.decision == "accept" and not warnings:
                        strength = "strong"
                        wording_policy = "assert"
                    elif decision.decision == "weaken":
                        wording_policy = "flag_uncertain"
            title, body = self._insight_text(aspect, claims, sorted(set(warnings)))
            insights.append(
                Insight(
                    insight_id=f"in_{index:04d}",
                    title=title,
                    body=body,
                    claim_ids=[claim.claim_id for claim in claims],
                    citation_spans=list(dict.fromkeys(citation_spans))[:8],
                    counter_evidence_spans=list(dict.fromkeys(counter_spans))[:6],
                    strength=strength,
                    wording_policy=wording_policy,
                    quality_warnings=sorted(set(warnings)),
                    freshness={
                        "newest_published_at": freshness_summary.newest_published_at,
                        "median_age_hours": freshness_summary.median_age_hours,
                    },
                )
            )

        graph.insights = insights
        return graph, self._markdown(graph, claim_by_id)


    @staticmethod
    def _aspect_sort_key(aspect: str, claims: List, graph: EvidenceGraph) -> Tuple[int, int, int, str]:
        query_text = " ".join(
            [task.query for task in graph.retrieval_tasks]
            + [variant for task in graph.retrieval_tasks for variant in task.query_variants]
        ).lower()
        primary_priority = 50
        if any(token in query_text for token in ["pricing", "price", "cost", "fee", "fees", "价格", "定价", "收费"]):
            if aspect == "pricing":
                primary_priority = 0
            elif aspect in {"usage_help", "general_discourse", "evidence_quality"}:
                primary_priority = 20
        support_count = sum(1 for claim in claims if claim.stance in {"support", "official"})
        oppose_count = sum(1 for claim in claims if claim.stance == "oppose")
        supported_count = sum(1 for claim in claims if claim.status == "supported")
        return (primary_priority, -supported_count, -(support_count + oppose_count), aspect)

    @staticmethod
    def _insight_text(aspect: str, claims: List, warnings: List[str]) -> Tuple[str, str]:
        readable = aspect.replace("_", " ")
        support_count = sum(1 for claim in claims if claim.stance in {"support", "official"})
        oppose_count = sum(1 for claim in claims if claim.stance == "oppose")
        if aspect == "pricing" and oppose_count and support_count:
            if support_count > oppose_count:
                title = "Pricing looks debated, not broadly rejected"
            else:
                title = "Negative pricing reaction is the main signal"
        elif aspect == "pricing" and support_count:
            title = "Pricing is mainly an official price-table story"
        elif aspect == "pricing" and oppose_count:
            title = "Negative pricing reaction is visible"
        elif oppose_count and support_count:
            title = f"{readable.title()} evidence is mixed across sampled sources"
        elif oppose_count:
            title = f"Negative {readable} signals appear in the sample"
        elif support_count:
            title = f"Supportive {readable} signals appear in the sample"
        else:
            title = f"{readable.title()} evidence is sample-limited"
        claim_suffix = "" if len(claims) == 1 else "s"
        if aspect == "pricing":
            body_parts = [
                f"{support_count} support/official vs {oppose_count} negative pricing signal{claim_suffix}",
                "criticism clusters around peak-hour or token-cost complaints",
            ]
        else:
            body_parts = [
                f"{len(claims)} {readable} claim group{claim_suffix}",
                f"{support_count} support/official",
                f"{oppose_count} negative",
            ]
        if "high_copy_ratio" in warnings:
            body_parts.append("repeated wording treated as amplification")
        if "ugc_only" in warnings or "single_source" in warnings:
            body_parts.append("sample-bound wording required")
        if any(claim.contradicting_spans for claim in claims):
            body_parts.append("counter-evidence retained")
        return title, "; ".join(body_parts) + "."

    @staticmethod
    def _markdown(graph: EvidenceGraph, claim_by_id: Dict) -> str:
        lines = ["# Signal Intelligence Synthesis", ""]
        if not graph.insights:
            lines.append("No final insight passed the source-span audit.")
            return "\n".join(lines)
        for insight in graph.insights:
            lines.append(f"## {insight.title}")
            lines.append(insight.body)
            lines.append(f"Cited sources: {len(insight.citation_spans)}.")
            if insight.counter_evidence_spans:
                lines.append(f"Counter-evidence sources: {len(insight.counter_evidence_spans)}.")
            if insight.quality_warnings:
                lines.append(f"Quality notes: {len(insight.quality_warnings)}.")
            lines.append("")
        return "\n".join(lines).strip()
