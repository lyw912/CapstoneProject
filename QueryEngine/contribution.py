"""Native QueryEngine projection into the shared evidence contracts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from AgentCoordinator.intelligence.contracts import (
    AcquisitionObservation,
    ClaimProposal,
    CoverageAssessment,
    EvidenceCandidate,
    EvidenceSpan,
    QueryContribution,
    ResearchTask,
    utc_now,
)
from AgentCoordinator.intelligence.evidence_core.blackboard import canonicalize_url, stable_id


def build_query_contribution(output: Dict[str, Any], task: ResearchTask) -> QueryContribution:
    sources: List[EvidenceCandidate] = []
    acquisitions: List[AcquisitionObservation] = []
    spans: List[EvidenceSpan] = []
    source_to_span: Dict[str, str] = {}
    source_types: Counter[str] = Counter()

    source_rows = list(output.get("sources") or [])[: max(0, task.budget.max_sources)]
    for rank, row in enumerate(source_rows, start=1):
        url = str(row.get("url") or "")
        title = str(row.get("title") or "")
        text = str(row.get("full_content") or row.get("snippet") or title)
        original_source_id = str(row.get("source_id") or stable_id("qsrc", url, title, text[:160]))
        platform = str(row.get("platform") or "web")
        source_api = str(row.get("source_api") or "query_engine")
        source_type = _source_type(row)
        source_types[source_type] += 1
        sources.append(
            EvidenceCandidate(
                source_id=original_source_id,
                platform=platform,
                source_type=source_type,
                source_name=platform,
                source_item_id=original_source_id,
                url=url,
                canonical_url=canonicalize_url(url),
                title=title,
                text=text,
                language=_language(f"{title} {text}"),
                published_at=row.get("published_at"),
                metadata={
                    "trust_score": float(row.get("trust_score") or 0.0),
                    "stance": row.get("stance_label"),
                    "stance_confidence": float(row.get("stance_confidence") or 0.0),
                    "rrf_score": row.get("rrf_score"),
                    "source_table": row.get("_source_table"),
                    "source_keyword": row.get("_source_keyword"),
                },
            )
        )
        query = str(row.get("sub_query_ref") or task.query)
        observed_at = utc_now()
        acquisitions.append(
            AcquisitionObservation(
                observation_id=stable_id("qobs", task.task_id, original_source_id, query, str(rank)),
                source_id=original_source_id,
                task_id=task.task_id,
                agent="query_agent",
                query=query,
                provider=source_api,
                tool="stance_aware_search",
                observed_at=observed_at,
                retrieved_at=observed_at,
                rank=rank,
                score=_optional_float(row.get("rrf_score")),
                raw_ref=f"query-engine://{task.task_id}/{original_source_id}",
                metadata={
                    "source_table": row.get("_source_table"),
                    "source_keyword": row.get("_source_keyword"),
                    "target_stance": row.get("stance_label") or row.get("_target_stance"),
                },
            )
        )
        excerpt = " ".join(text.split())[:500]
        if excerpt:
            span_id = stable_id("qsp", original_source_id, excerpt)
            source_to_span[original_source_id] = span_id
            spans.append(
                EvidenceSpan(
                    span_id=span_id,
                    source_id=original_source_id,
                    text=excerpt,
                    start_char=0,
                    end_char=len(excerpt),
                    span_type="source_excerpt",
                    extraction_route="query_agent_output",
                    confidence=max(0.45, float(row.get("trust_score") or 0.0)),
                    locator={"url": url},
                )
            )

    social_sentiment = output.get("social_sentiment") or {}
    for rank, voice in enumerate(social_sentiment.get("top_social_voices") or [], start=1):
        if len(sources) >= max(0, task.budget.max_sources):
            break
        text = str(voice.get("content") or "").strip()
        if not text:
            continue
        platform = str(voice.get("platform") or "social")
        url = str(voice.get("url") or "")
        published_at = voice.get("publish_time")
        source_id = stable_id("qms", platform, url, text[:200], str(published_at or ""))
        source_types["ugc"] += 1
        sources.append(
            EvidenceCandidate(
                source_id=source_id,
                platform=platform,
                source_type="ugc",
                source_name=platform,
                source_item_id=source_id,
                url=url,
                canonical_url=canonicalize_url(url),
                title=f"{platform} public voice",
                text=text,
                language=_language(text),
                published_at=published_at,
                metadata={
                    "stance": voice.get("stance"),
                    "mindspider_mode": social_sentiment.get("mode"),
                },
            )
        )
        observed_at = utc_now()
        acquisitions.append(
            AcquisitionObservation(
                observation_id=stable_id("qmso", task.task_id, source_id, str(rank)),
                source_id=source_id,
                task_id=task.task_id,
                agent="query_agent",
                query=task.query,
                provider="mindspider_db",
                tool="social_enrichment",
                observed_at=observed_at,
                retrieved_at=observed_at,
                rank=rank,
                raw_ref=f"mindspider-contribution://{task.task_id}/{source_id}",
                metadata={"platform": platform},
            )
        )
        excerpt = " ".join(text.split())[:500]
        span_id = stable_id("qmssp", source_id, excerpt)
        spans.append(
            EvidenceSpan(
                span_id=span_id,
                source_id=source_id,
                text=excerpt,
                start_char=0,
                end_char=len(excerpt),
                span_type="social_voice_excerpt",
                extraction_route="query_social_enrichment",
                confidence=0.6,
                locator={"url": url, "platform": platform},
            )
        )
    proposals: List[ClaimProposal] = []
    for cluster in output.get("opinion_clusters") or []:
        claim_text = str(cluster.get("core_argument") or "").strip()
        if not claim_text:
            continue
        evidence_ids = [
            source_to_span[source_id]
            for source_id in cluster.get("evidence_sources") or []
            if source_id in source_to_span
        ]
        proposals.append(
            ClaimProposal(
                proposal_id=stable_id("qcp", task.task_id, claim_text),
                agent="query_agent",
                claim_text=claim_text,
                claim_type="stance",
                target_entity=task.query,
                aspect="public_discourse",
                stance=str(cluster.get("stance") or "neutral"),
                evidence_span_ids=evidence_ids,
                task_id=task.task_id,
                confidence=min(0.9, 0.45 + len(evidence_ids) * 0.08),
                uncertainty=[] if evidence_ids else ["no_addressable_source_span"],
            )
        )

    stance_counts = Counter(str(row.get("stance_label") or "unclassified") for row in source_rows)
    missing = list(output.get("knowledge_gaps") or [])
    for stance in ["official", "support", "oppose", "neutral"]:
        if stance_counts.get(stance, 0) == 0:
            missing.append(f"missing_stance:{stance}")
    coverage = CoverageAssessment(
        assessment_id=stable_id("qcov", task.task_id),
        task_id=task.task_id,
        agent="query_agent",
        score=float(output.get("coverage_score") or 0.0),
        stance_counts=dict(stance_counts),
        source_type_counts=dict(source_types),
        covered_dimensions=[key for key, count in stance_counts.items() if count > 0],
        missing_dimensions=list(dict.fromkeys(missing)),
        limitations=["Query coverage measures observable retrieved sources, not population opinion."],
    )
    runtime_errors = list(output.get("runtime_errors") or [])
    status = "complete" if sources and not runtime_errors else "partial"
    return QueryContribution(
        contribution_id=stable_id("qc", task.task_id, str(len(sources)), str(output.get("search_iterations") or 0)),
        task_id=task.task_id,
        agent="query_agent",
        status=status,
        sources=sources,
        acquisitions=acquisitions,
        evidence_spans=spans,
        claim_proposals=proposals,
        coverage=coverage,
        trace=list(output.get("runtime_trace") or output.get("trace_log") or []),
        errors=runtime_errors,
        stance_distribution=dict(output.get("stance_distribution") or {}),
        opinion_clusters=list(output.get("opinion_clusters") or []),
        knowledge_gaps=list(output.get("knowledge_gaps") or []),
        social_sentiment=output.get("social_sentiment"),
    )


def _source_type(row: Dict[str, Any]) -> str:
    stance = str(row.get("stance_label") or "")
    platform = str(row.get("platform") or "").lower()
    source_api = str(row.get("source_api") or "").lower()
    if stance == "official":
        return "official"
    if source_api == "mindspider_db" or platform in {"weibo", "zhihu", "bilibili", "douyin", "kuaishou", "xiaohongshu", "tieba"}:
        return "ugc"
    return "search_result"


def _language(text: str) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"


def _optional_float(value: Any):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
