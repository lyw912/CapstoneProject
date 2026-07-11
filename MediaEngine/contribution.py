"""Native MediaEngine projection into section dossiers and evidence batches."""

from __future__ import annotations

from collections import Counter
from typing import List

from AgentCoordinator.intelligence.contracts import (
    AcquisitionObservation,
    CoverageAssessment,
    EvidenceCandidate,
    EvidenceSpan,
    MediaContribution,
    ResearchTask,
    SectionDossier,
)
from AgentCoordinator.intelligence.evidence_core.blackboard import canonicalize_url, stable_id


def build_media_contribution(state, task: ResearchTask, graph_state=None) -> MediaContribution:
    sources: List[EvidenceCandidate] = []
    acquisitions: List[AcquisitionObservation] = []
    spans: List[EvidenceSpan] = []
    dossiers: List[SectionDossier] = []
    source_types: Counter[str] = Counter()
    seen_source_ids = set()
    max_sources = max(0, task.budget.max_sources)

    for paragraph in state.paragraphs:
        dossier_source_ids: List[str] = []
        dossier_span_ids: List[str] = []
        assets = []
        for rank, search in enumerate(paragraph.research.search_history, start=1):
            if not search.has_result:
                continue
            source_id = stable_id("msrc", search.url, search.title, search.content[:160])
            if source_id not in seen_source_ids and len(seen_source_ids) >= max_sources:
                continue
            source_type = "image" if search.result_type == "image" else "media_search_result"
            source_types[source_type] += 1
            if source_id not in seen_source_ids:
                seen_source_ids.add(source_id)
                sources.append(
                    EvidenceCandidate(
                        source_id=source_id,
                        platform=_platform(search.url),
                        source_type=source_type,
                        source_name=_platform(search.url),
                        source_item_id=source_id,
                        url=search.url,
                        canonical_url=canonicalize_url(search.url),
                        title=search.title,
                        text=search.content,
                        language=_language(f"{search.title} {search.content}"),
                        published_at=search.published_at,
                        metadata={"result_type": search.result_type, **dict(search.metadata)},
                    )
                )
            observation_id = stable_id("mobs", task.task_id, paragraph.title, search.query, source_id, str(rank))
            acquisitions.append(
                AcquisitionObservation(
                    observation_id=observation_id,
                    source_id=source_id,
                    task_id=task.task_id,
                    agent="media_agent",
                    query=search.query,
                    provider="media_engine",
                    tool=search.search_tool or "multimodal_search",
                    observed_at=search.timestamp,
                    retrieved_at=search.timestamp,
                    rank=rank,
                    score=search.score,
                    raw_ref=f"media-engine://{task.task_id}/{paragraph.order}/{rank}",
                    metadata={"section_id": str(paragraph.order), "result_type": search.result_type},
                )
            )
            excerpt = " ".join(search.content.split())[:500]
            if excerpt:
                span_id = stable_id("msp", source_id, paragraph.title, excerpt)
                spans.append(
                    EvidenceSpan(
                        span_id=span_id,
                        source_id=source_id,
                        text=excerpt,
                        start_char=0,
                        end_char=len(excerpt),
                        span_type="media_excerpt",
                        modality="image" if search.result_type == "image" else "text",
                        locator={"url": search.url, "section": paragraph.title},
                        extraction_route="media_paragraph_research",
                        confidence=max(0.45, float(search.score or 0.0)),
                    )
                )
                dossier_span_ids.append(span_id)
            dossier_source_ids.append(source_id)
            if search.result_type == "image" or search.image_url:
                assets.append(
                    {
                        "type": "image",
                        "url": search.image_url or search.url,
                        "title": search.title,
                        "source_id": source_id,
                    }
                )

        summary = paragraph.research.latest_summary or ""
        dossiers.append(
            SectionDossier(
                dossier_id=stable_id("dos", task.task_id, str(paragraph.order), paragraph.title),
                task_id=task.task_id,
                section_id=str(paragraph.order),
                title=paragraph.title,
                objective=paragraph.content,
                summary=summary,
                source_ids=list(dict.fromkeys(dossier_source_ids)),
                evidence_span_ids=list(dict.fromkeys(dossier_span_ids)),
                multimodal_assets=assets,
                unresolved_questions=[] if summary else ["section_summary_missing"],
                reflection_rounds=paragraph.research.reflection_iteration,
                status="complete" if paragraph.is_completed() else "partial",
            )
        )

    completed = sum(1 for dossier in dossiers if dossier.status == "complete")
    coverage_score = completed / len(dossiers) if dossiers else 0.0
    missing = [f"incomplete_section:{dossier.title}" for dossier in dossiers if dossier.status != "complete"]
    coverage = CoverageAssessment(
        assessment_id=stable_id("mcov", task.task_id),
        task_id=task.task_id,
        agent="media_agent",
        score=round(coverage_score, 4),
        source_type_counts=dict(source_types),
        covered_dimensions=[dossier.title for dossier in dossiers if dossier.status == "complete"],
        missing_dimensions=missing,
        limitations=["Media dossiers describe retrieved material and do not independently validate claims."],
    )
    runtime_errors = list((graph_state or {}).get("error_log") or [])
    return MediaContribution(
        contribution_id=stable_id("mc", task.task_id, str(len(sources)), str(len(dossiers))),
        task_id=task.task_id,
        agent="media_agent",
        status="complete" if dossiers and not missing and not runtime_errors else "partial",
        sources=sources,
        acquisitions=acquisitions,
        evidence_spans=spans,
        coverage=coverage,
        trace=list((graph_state or {}).get("trace_log") or [])
        or [f"section:{dossier.title}:{dossier.status}" for dossier in dossiers],
        errors=runtime_errors,
        dossiers=dossiers,
        narrative_summary=state.final_report or "",
    )


def _platform(url: str) -> str:
    try:
        from urllib.parse import urlsplit

        return urlsplit(url).netloc.lower() or "media"
    except ValueError:
        return "media"


def _language(text: str) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"
