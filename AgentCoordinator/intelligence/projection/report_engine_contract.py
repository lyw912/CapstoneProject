"""Projection from the Coordinator intelligence ledger to coordinator_output_latest.json."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from AgentCoordinator.utils.platform_profiles import PLATFORM_PROFILES, SOCIAL_PLATFORM_KEYS, canonical_social_platform

from ..contracts import EvidenceGraph, QualityFeatures, CoordinatorIntelligenceArtifact

SOCIAL_PLATFORMS = SOCIAL_PLATFORM_KEYS


def build_coordinator_output_from_artifact(artifact: CoordinatorIntelligenceArtifact, duration_seconds: float) -> Dict[str, Any]:
    graph = artifact.evidence_graph
    quality_by_item = graph.quality_index()
    top_sources = _top_sources(graph, quality_by_item)
    divergence = _divergence_matrix(graph, quality_by_item)
    synthesis = _synthesis(artifact)
    source_data = _source_data(artifact, graph, quality_by_item, top_sources, divergence)
    fact_opinion = _fact_opinion_separation(graph, quality_by_item)
    bias = _bias_analysis(artifact, graph, quality_by_item)

    return {
        "schema_version": "2.1-coordinator-intelligence",
        "query": artifact.query,
        "analysis_type": artifact.analysis_type,
        "generated_at": artifact.created_at,
        "pipeline_duration_seconds": round(float(duration_seconds), 2),
        "artifact_derivation": {
            "primary_record": "coordinator_intelligence",
            "compatibility_views": [
                "synthesis",
                "source_data",
                "divergence_matrix",
                "deliberation",
                "gap_filling",
                "platform_interpretations",
                "bias_analysis",
                "fact_opinion_separation",
            ],
            "principle": (
                "Compatibility fields are evidence-derived Coordinator views, "
                "not independent second-pass conclusions."
            ),
        },
        "coordinator_intelligence": artifact.to_dict(),
        "divergence_matrix": divergence,
        "deliberation": {
            "analysis_type": artifact.analysis_type,
            "perspectives_used": _perspectives_used(artifact.analysis_type),
            "phases": [
                {
                    "phase": "claim_level_audit",
                    "summary": (
                        "Claim-level review combines specialist proposals, "
                        "skeptical counter-evidence, methodological quality checks, and judge decisions "
                        "all point back to cited source spans."
                    ),
                    "consensus_points": [insight.title for insight in artifact.insights[:5]],
                    "dissent_points": [edge.explanation for edge in graph.contradiction_edges[:5]],
                    "audit_decisions": [decision.to_dict() for decision in graph.audit_decisions[:12]],
                }
            ],
            "final_consensus": [insight.title for insight in artifact.insights[:5]],
            "final_dissents": [edge.explanation for edge in graph.contradiction_edges[:5]],
            "confidence": synthesis["overall_confidence"],
        },
        "gap_filling": {
            "rounds_performed": _research_rounds(graph),
            "gaps_detected": [
                {
                    "description": f"{task.purpose} search for claim {task.parent_claim_id or 'initial'}",
                    "source": task.target_source,
                    "query": task.query,
                }
                for task in graph.retrieval_tasks
                if task.parent_claim_id
            ]
            + [
                {
                    "description": task.objective,
                    "source": task.agent,
                    "query": task.query,
                }
                for task in graph.research_tasks
                if task.round_index > 0
            ],
            "results_found": max(
                sum(result.items_returned for result in graph.retrieval_results),
                len(graph.acquisition_observations),
            ),
        },
        "platform_interpretations": _platform_interpretations(graph, quality_by_item),
        "bias_analysis": bias,
        "fact_opinion_separation": fact_opinion,
        "synthesis": synthesis,
        "source_data": source_data,
        "coordinator_trace": [
            f"[{step.node}] {step.route} input={step.input_count} output={step.output_count} elapsed={step.elapsed_ms}ms"
            for step in artifact.research_trace
        ],
        "agent_errors": [
            error
            for diagnostic in artifact.provider_diagnostics
            for error in diagnostic.errors
        ],
    }


def _synthesis(artifact: CoordinatorIntelligenceArtifact) -> Dict[str, Any]:
    insights = [
        {
            "insight": insight.title,
            "basis": insight.body,
            "confidence": _insight_confidence(insight.strength),
            "claim_ids": insight.claim_ids,
            "citation_spans": insight.citation_spans,
            "quality_warnings": insight.quality_warnings,
            "wording_policy": insight.wording_policy,
        }
        for insight in artifact.insights
    ]
    confidence = sum(item["confidence"] for item in insights) / len(insights) if insights else 0.0
    return {
        "summary": (
            insights[0]["insight"]
            if insights
            else "No final insight passed the evidence audit."
        ),
        "top_insights": insights,
        "key_tensions": _key_tensions(artifact.evidence_graph),
        "overall_confidence": round(confidence, 4),
        "recommended_investigation": _recommendations(artifact),
    }


def _top_sources(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> List[Dict[str, Any]]:
    rows = []
    for evidence in graph.evidence_items:
        quality = quality_by_item.get(evidence.item_id)
        if not quality:
            continue
        rows.append(
            {
                "title": evidence.title or evidence.source_name or evidence.url,
                "url": evidence.url,
                "trust_score": round(max(quality.persuasiveness_score, quality.source_authority_score * 0.85), 4),
                "stance": quality.stance,
                "sentiment": quality.sentiment,
                "platform": evidence.platform,
                "source_type": evidence.source_type,
                "canonical_item_id": evidence.canonical_item_id,
                "citation_span_id": evidence.spans[0].span_id if evidence.spans else "",
                "quality_warnings": quality.low_quality_reasons,
                "copy_ratio": quality.copy_ratio_in_cluster,
                "acquisition_source": evidence.acquisition_source,
            }
        )
    return sorted(rows, key=lambda item: item["trust_score"], reverse=True)[:12]


def _source_data(
    artifact: CoordinatorIntelligenceArtifact,
    graph: EvidenceGraph,
    quality_by_item: Dict[str, QualityFeatures],
    top_sources: List[Dict[str, Any]],
    divergence: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the legacy source_data contract as a view over the evidence ledger."""
    social_sentiment = _social_sentiment(graph, quality_by_item, divergence)
    dossiers = list(graph.section_dossiers)
    multimodal_assets = sum(len(item.multimodal_assets) for item in dossiers)
    media_runs = [item for item in graph.agent_runs if item.agent == "media_agent"]
    media_errors = [error for item in media_runs for error in item.errors]
    media_available = bool(dossiers or any(item.status == "complete" for item in media_runs))
    query_coverage = [item for item in graph.coverage_assessments if item.agent == "query_agent"]
    query_coverage_score = query_coverage[-1].score if query_coverage else _coverage_score(artifact)
    return {
        "query_agent": {
            "derived_from": "coordinator_intelligence.evidence_graph",
            "total_sources": artifact.evidence_graph_summary.get("raw_count", 0),
            "total_sources_found": artifact.evidence_graph_summary.get("raw_count", 0),
            "total_sources_kept": artifact.evidence_graph_summary.get("canonical_count", 0),
            "canonical_sources": artifact.evidence_graph_summary.get("canonical_count", 0),
            "stance_distribution": _stance_distribution(graph, quality_by_item),
            "coverage_score": query_coverage_score,
            "coverage_assessments": [item.to_dict() for item in query_coverage],
            "acquisition_observations": sum(
                1 for item in graph.acquisition_observations if item.agent == "query_agent"
            ),
            "top_sources": top_sources,
            "opinion_clusters": _opinion_clusters(graph, quality_by_item),
            "knowledge_gaps": _knowledge_gaps(artifact),
            "quality_summary": artifact.quality_summary,
            "freshness_summary": artifact.freshness_summary.to_dict(),
            "source_coverage": artifact.source_coverage,
            "social_sentiment": social_sentiment,
        },
        "media_agent": {
            "available": media_available,
            "mode": "live" if media_available else "failed_or_unavailable",
            "summary_length": sum(len(item.summary) for item in dossiers),
            "section_dossiers": len(dossiers),
            "completed_dossiers": sum(1 for item in dossiers if item.status == "complete"),
            "multimodal_assets": multimodal_assets,
            "source_observations": sum(
                1 for item in graph.acquisition_observations if item.agent == "media_agent"
            ),
            "dossiers": [
                {
                    "section_id": item.section_id,
                    "title": item.title,
                    "objective": item.objective,
                    "summary": item.summary,
                    "status": item.status,
                    "source_ids": item.source_ids,
                    "evidence_span_ids": item.evidence_span_ids,
                    "multimodal_asset_count": len(item.multimodal_assets),
                    "unresolved_questions": item.unresolved_questions,
                }
                for item in dossiers
            ],
            "errors": media_errors,
            "note": (
                "MediaAgent contributes source-bound section dossiers and multimodal observations; "
                "ReportEngine remains the sole final-document renderer."
            ),
        },
    }


def _social_sentiment(
    graph: EvidenceGraph,
    quality_by_item: Dict[str, QualityFeatures],
    divergence: Dict[str, Any],
) -> Dict[str, Any]:
    social_items = [
        item
        for item in graph.normalized_items
        if item.source_type in {"ugc", "comment"} and canonical_social_platform(item.platform)
    ]
    per_platform: Dict[str, Dict[str, Any]] = {}
    for item in social_items:
        quality = quality_by_item.get(item.item_id)
        if not quality:
            continue
        platform_key = canonical_social_platform(item.platform) or item.platform
        bucket = per_platform.setdefault(platform_key, {"count": 0, "distribution": {}})
        bucket["count"] += 1
        bucket["distribution"][quality.stance] = bucket["distribution"].get(quality.stance, 0) + 1
    for bucket in per_platform.values():
        total = bucket["count"] or 1
        bucket["post_count"] = bucket["count"]
        bucket["distribution"] = {
            stance: round(count / total, 4)
            for stance, count in sorted(bucket["distribution"].items(), key=lambda item: -item[1])
        }

    total_posts = len(social_items)
    content_texts = [item.text for item in social_items if item.text]
    diversity = round(len(set(content_texts)) / max(1, len(content_texts)), 4) if content_texts else 0.0
    low_diversity_warning = None
    if total_posts >= 5 and diversity < 0.7:
        low_diversity_warning = (
            f"Low content diversity ({diversity:.0%}) suggests repeated platform samples; "
            "treat this as amplification rather than independent viewpoint prevalence."
        )
    mode = "available" if total_posts else "disabled"
    return {
        "mode": mode,
        "derived_from": "coordinator_intelligence.evidence_graph.normalized_items",
        "platforms_queried": sorted(per_platform),
        "total_posts": total_posts,
        "total_comments": sum(1 for item in social_items if item.source_type == "comment"),
        "sentiment_distribution": _sentiment_distribution_for_items(social_items, quality_by_item),
        "per_platform": per_platform,
        "content_diversity": diversity,
        "low_diversity_warning": low_diversity_warning,
        "divergence_score": (divergence.get("max_divergence") or {}).get("value", 0.0),
        "divergence_summary": _divergence_summary(divergence),
        "freshness_hours": None,
        "top_social_voices": [
            {
                "platform": item.platform,
                "stance": quality_by_item[item.item_id].stance,
                "content": item.text[:260],
                "url": item.url,
                "publish_time": item.published_at,
            }
            for item in social_items
            if item.item_id in quality_by_item
        ][:10],
        "comment_sentiment": None,
        "sentiment_trend": None,
        "crawl_triggered": False,
    }


def _opinion_clusters(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> List[Dict[str, Any]]:
    by_stance: Dict[str, List[Any]] = {}
    for claim in graph.claims:
        if claim.status in {"unsupported", "demoted"}:
            continue
        by_stance.setdefault(claim.stance or "neutral", []).append(claim)
    total = sum(len(items) for items in by_stance.values()) or 1
    clusters = []
    for stance, claims in sorted(by_stance.items(), key=lambda item: -len(item[1])):
        clusters.append(
            {
                "stance": stance,
                "core_argument": " ".join(claim.claim_text for claim in claims[:2]),
                "source_count": len(set(span for claim in claims for span in claim.supporting_spans)),
                "estimated_proportion": round(len(claims) / total, 4),
                "claim_ids": [claim.claim_id for claim in claims[:8]],
            }
        )
    return clusters


def _knowledge_gaps(artifact: CoordinatorIntelligenceArtifact) -> List[str]:
    gaps = []
    if artifact.audit_summary.get("needs_search_but_budget_exhausted", 0):
        gaps.append("Some claims still need follow-up retrieval after the configured research budget.")
    if not artifact.source_coverage.get("platform_sources"):
        gaps.append("No observable social-platform samples were available in this run; platform interpretation is limited.")
    if any(item.status in {"not_configured", "error"} for item in artifact.provider_diagnostics):
        gaps.append("Provider diagnostics include missing or failed optional providers.")
    return gaps[:5]


def _stance_distribution(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    total = 0.0
    for cluster in graph.canonical_clusters:
        quality = quality_by_item.get(cluster.representative_item_id)
        if not quality:
            continue
        weight = 1.0 + math.log1p(max(0, cluster.amplification_count - 1)) * 0.25
        weight *= max(0.2, 1.0 - quality.copy_ratio_in_cluster * 0.55)
        weights[quality.stance] = weights.get(quality.stance, 0.0) + weight
        total += weight
    if total <= 0:
        return {}
    return {stance: round(value / total, 4) for stance, value in sorted(weights.items(), key=lambda item: -item[1])}


def _sentiment_distribution(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    total = 0.0
    for cluster in graph.canonical_clusters:
        quality = quality_by_item.get(cluster.representative_item_id)
        if not quality:
            continue
        weight = 1.0 + math.log1p(max(0, cluster.amplification_count - 1)) * 0.15
        weights[quality.sentiment] = weights.get(quality.sentiment, 0.0) + weight
        total += weight
    if total <= 0:
        return {}
    return {sentiment: round(value / total, 4) for sentiment, value in sorted(weights.items(), key=lambda item: -item[1])}


def _sentiment_distribution_for_items(items: List[Any], quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items:
        quality = quality_by_item.get(item.item_id)
        if not quality:
            continue
        weights[quality.stance] = weights.get(quality.stance, 0.0) + 1.0
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {label: round(value / total, 4) for label, value in sorted(weights.items(), key=lambda item: -item[1])}


def _divergence_matrix(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, Any]:
    platform_stance: Dict[str, Dict[str, float]] = {}
    for cluster in graph.canonical_clusters:
        quality = quality_by_item.get(cluster.representative_item_id)
        if not quality:
            continue
        for platform in cluster.platforms:
            bucket = platform_stance.setdefault(platform, {})
            bucket[quality.stance] = bucket.get(quality.stance, 0.0) + 1.0
    normalized = {}
    for platform, dist in platform_stance.items():
        total = sum(dist.values()) or 1.0
        normalized[platform] = {stance: value / total for stance, value in dist.items()}
    pairs = {}
    platforms = sorted(normalized)
    for idx, left in enumerate(platforms):
        for right in platforms[idx + 1 :]:
            pairs[f"{left}|{right}"] = round(_distribution_distance(normalized[left], normalized[right]), 4)
    hotspots = [
        f"{pair} shows stance-distribution divergence of {value:.2f}"
        for pair, value in pairs.items()
        if value >= 0.30
    ]
    max_pair = max(pairs.items(), key=lambda item: item[1]) if pairs else ("N/A", 0.0)
    min_pair = min(pairs.items(), key=lambda item: item[1]) if pairs else ("N/A", 0.0)
    return {
        "pairs": pairs,
        "hotspots": hotspots,
        "max_divergence": {"pair": max_pair[0], "value": max_pair[1]},
        "min_divergence": {"pair": min_pair[0], "value": min_pair[1]},
        "method": "CSSD over stance distributions derived from Coordinator evidence clusters",
    }


def _divergence_summary(divergence: Dict[str, Any]) -> str:
    max_div = divergence.get("max_divergence") or {}
    pair = max_div.get("pair")
    value = max_div.get("value", 0)
    if not pair or pair == "N/A":
        return "No cross-source divergence could be computed from the available evidence groups."
    return f"The largest evidence-derived stance divergence is {pair} at CSSD={float(value):.2f}."


def _distribution_distance(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2.0


def _key_tensions(graph: EvidenceGraph) -> List[Dict[str, Any]]:
    claim_by_id = {claim.claim_id: claim for claim in graph.claims}
    grouped: Dict[str, Dict[str, Any]] = {}
    for edge in graph.contradiction_edges:
        left = claim_by_id.get(edge.claim_a)
        aspect = getattr(left, "aspect", "evidence") or "evidence"
        readable = str(aspect).replace("_", " ")
        key = str(aspect)
        row = grouped.setdefault(
            key,
            {
                "tension": f"Mixed {readable} signals",
                "between": [],
                "significance": "Review cited sources before making a broad conclusion.",
                "conflict_count": 0,
            },
        )
        row["conflict_count"] += 1
        for claim_id in [edge.claim_a, edge.claim_b]:
            if claim_id not in row["between"] and len(row["between"]) < 6:
                row["between"].append(claim_id)
    return sorted(grouped.values(), key=lambda item: item.get("conflict_count", 0), reverse=True)[:6]


def _verified_facts(graph: EvidenceGraph) -> List[Dict[str, Any]]:
    span_index = graph.span_index()
    evidence_by_id = {item.evidence_id: item for item in graph.evidence_items}
    rows = []
    for claim in graph.claims:
        if claim.status != "supported" or not claim.supporting_spans:
            continue
        sources = []
        for span_id in claim.supporting_spans:
            span = span_index.get(span_id)
            if span:
                evidence = evidence_by_id.get(span.evidence_id)
                if evidence and evidence.url:
                    sources.append(evidence.url)
        rows.append(
            {
                "fact": claim.claim_text,
                "sources": list(dict.fromkeys(sources)),
                "source_spans": claim.supporting_spans,
                "verification_status": claim.status,
                "confidence": claim.confidence,
            }
        )
    return rows[:12]


def _fact_opinion_separation(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, Any]:
    return {
        "verified_facts": _verified_facts(graph),
        "opinions_sentiments": _opinions(graph, quality_by_item),
        "analytical_frameworks": [
            {
                "framework": "EvidenceGraph audit",
                "analysis": "Every final insight points to claim ids and source span ids.",
                "certainty": "high",
            },
            {
                "framework": "Repeated-coverage weighting",
                "analysis": "Repeated posts affect coverage strength but are not counted as independent viewpoints.",
                "certainty": "high",
            },
            {
                "framework": "Platform-aware interpretation",
                "analysis": "Social-platform samples are interpreted with platform context only when observable platform samples are present.",
                "certainty": "medium",
            },
        ],
    }


def _opinions(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> List[Dict[str, Any]]:
    rows = []
    for evidence in graph.evidence_items[:12]:
        quality = quality_by_item.get(evidence.item_id)
        if not quality:
            continue
        rows.append(
            {
                "perspective": quality.stance,
                "holders": evidence.platform,
                "sentiment_intensity": quality.sentiment,
                "potential_biases": quality.low_quality_reasons,
                "source_span_ids": [span.span_id for span in evidence.spans],
            }
        )
    return rows


def _platform_interpretations(graph: EvidenceGraph, quality_by_item: Dict[str, QualityFeatures]) -> Dict[str, str]:
    by_platform: Dict[str, List[str]] = {}
    for evidence in graph.evidence_items:
        quality = quality_by_item.get(evidence.item_id)
        if not quality:
            continue
        platform_key = canonical_social_platform(evidence.platform)
        if not platform_key or evidence.source_type not in {"ugc", "comment"}:
            continue
        by_platform.setdefault(platform_key, []).append(f"{quality.stance}/{quality.sentiment}")
    rows: Dict[str, str] = {}
    for platform, labels in sorted(by_platform.items()):
        label_text = ", ".join(sorted(set(labels)))
        profile = PLATFORM_PROFILES.get(platform, {})
        display_name = profile.get("display_name", platform)
        demographic_note = profile.get("demographic_note")
        bias_note = profile.get("bias_tendency")
        rows[platform] = (
            f"{display_name} contributed {len(labels)} observable platform sample(s). "
            f"Observed stance and sentiment labels include {label_text}."
            + (f" Demographic context: {demographic_note}." if demographic_note else "")
            + (f" Interpretation caution: {bias_note}." if bias_note else "")
        )
    return rows


def _bias_analysis(
    artifact: CoordinatorIntelligenceArtifact,
    graph: EvidenceGraph,
    quality_by_item: Dict[str, QualityFeatures],
) -> Dict[str, Any]:
    warnings = list(artifact.quality_summary.get("quality_warnings", []) or [])
    platform_distributions: Dict[str, Dict[str, float]] = {}
    for evidence in graph.evidence_items:
        quality = quality_by_item.get(evidence.item_id)
        if not quality:
            continue
        bucket = platform_distributions.setdefault(evidence.platform, {})
        bucket[quality.stance] = bucket.get(quality.stance, 0.0) + 1.0
    for platform, counts in platform_distributions.items():
        total = sum(counts.values()) or 1.0
        dist = {stance: count / total for stance, count in counts.items()}
        entropy = _shannon_entropy(dist)
        dominant = max(dist, key=dist.get) if dist else None
        if dominant and total >= 3 and entropy < 0.5 and dist[dominant] > 0.7:
            warnings.append(
                f"{platform}: low stance entropy ({entropy:.2f}) with dominant {dominant} signal; possible echo-chamber effect."
            )
    social = {}
    for platform, dist in platform_distributions.items():
        platform_key = canonical_social_platform(platform)
        if platform_key:
            social[platform_key] = dist
    silent_hypothesis = None
    if not social:
        silent_hypothesis = "No observable social-platform samples were returned, so silent-majority inference is not attempted."
    else:
        silent_hypothesis = "Observable platform samples cannot be converted into population-level silent-majority claims."
    return {
        "echo_warnings": list(dict.fromkeys(warnings)),
        "silent_majority_hypothesis": silent_hypothesis,
    }


def _shannon_entropy(distribution: Dict[str, float]) -> float:
    entropy = 0.0
    for value in distribution.values():
        if value > 1e-9:
            entropy -= value * math.log2(value)
    return round(entropy, 4)


def _coverage_score(artifact: CoordinatorIntelligenceArtifact) -> float:
    summary = artifact.evidence_graph_summary
    canonical = float(summary.get("canonical_count") or 0)
    supported = float(summary.get("supported_claims") or 0)
    diagnostics_penalty = 0.08 * sum(1 for item in artifact.provider_diagnostics if item.status in {"error", "not_configured"})
    return round(max(0.0, min(1.0, 0.25 + canonical * 0.06 + supported * 0.05 - diagnostics_penalty)), 4)


def _research_rounds(graph: EvidenceGraph) -> int:
    fusion_round = max((task.round_index for task in graph.research_tasks), default=0)
    legacy_claim_rounds = {task.parent_claim_id for task in graph.retrieval_tasks if task.parent_claim_id}
    return max(fusion_round, len(legacy_claim_rounds))


def _recommendations(artifact: CoordinatorIntelligenceArtifact) -> List[str]:
    recs = []
    semantic = [
        item
        for item in artifact.provider_diagnostics
        if item.provider == "jina" and item.capability in {"embedding_and_rerank", "embedding", "rerank"}
    ]
    if semantic and all(item.status == "not_configured" for item in semantic):
        recs.append("Configure Jina for semantic duplicate clustering and rerank when provider-backed quality scoring is required.")
    if any(item.provider == "jina" and item.status == "error" for item in semantic):
        recs.append("Review semantic provider diagnostics and rerun when the configured provider is reachable.")
    if any(item.status == "error" and item.provider != "jina" for item in artifact.provider_diagnostics):
        recs.append("Review provider diagnostics and rerun when the failing search provider is reachable.")
    if artifact.quality_summary.get("coordination_warning"):
        recs.append("Treat repeated wording as amplification, not independent agreement.")
    if not recs:
        recs.append("Review the cited source excerpts before publishing external conclusions.")
    return recs


def _perspectives_used(analysis_type: str) -> List[str]:
    return [
        "Query breadth and stance specialist",
        "Media narrative and multimodal specialist",
        "Evidence-bound proposer",
        "Counter-evidence skeptic",
        "Evidence methodologist",
        "Claim judge",
    ]


def _insight_confidence(strength: str) -> float:
    return {"strong": 0.86, "moderate": 0.68, "weak": 0.48, "uncertain": 0.32}.get(strength, 0.5)
