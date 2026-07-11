"""Append-only, single-reducer evidence blackboard.

Specialists submit immutable contribution batches. Only this reducer canonicalizes
sources and advances the blackboard version, which avoids parallel state writes
and preserves every acquisition route independently from the source entity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..contracts import (
    AcquisitionObservation,
    AgentContribution,
    AgentRunRecord,
    ClaimProposal,
    CoverageAssessment,
    EvidenceCandidate,
    EvidenceSpan,
    MediaContribution,
    NormalizedItem,
    ResearchTask,
    SectionDossier,
    utc_now,
)


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "spm",
}


def stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8', errors='ignore')).hexdigest()[:16]}"


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        filtered = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
        ]
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(filtered), ""))
    except ValueError:
        return raw.split("#", 1)[0]


@dataclass(frozen=True)
class BlackboardEvent:
    version: int
    event_type: str
    entity_id: str
    contribution_id: str
    recorded_at: str


@dataclass
class EvidenceBlackboardSnapshot:
    version: int
    sources: List[EvidenceCandidate]
    acquisitions: List[AcquisitionObservation]
    evidence_spans: List[EvidenceSpan]
    claim_proposals: List[ClaimProposal]
    coverage_assessments: List[CoverageAssessment]
    section_dossiers: List[SectionDossier]
    research_tasks: List[ResearchTask]
    agent_runs: List[AgentRunRecord]
    events: List[BlackboardEvent]


@dataclass
class EvidenceBlackboard:
    """In-memory run-scoped blackboard with deterministic, idempotent ingest."""

    run_id: str
    version: int = 0
    _sources: Dict[str, EvidenceCandidate] = field(default_factory=dict)
    _source_aliases: Dict[str, str] = field(default_factory=dict)
    _acquisitions: Dict[str, AcquisitionObservation] = field(default_factory=dict)
    _spans: Dict[str, EvidenceSpan] = field(default_factory=dict)
    _proposals: Dict[str, ClaimProposal] = field(default_factory=dict)
    _coverage: Dict[str, CoverageAssessment] = field(default_factory=dict)
    _dossiers: Dict[str, SectionDossier] = field(default_factory=dict)
    _tasks: Dict[str, ResearchTask] = field(default_factory=dict)
    _agent_runs: Dict[str, AgentRunRecord] = field(default_factory=dict)
    _contributions: set[str] = field(default_factory=set)
    _events: List[BlackboardEvent] = field(default_factory=list)

    def register_tasks(self, tasks: Iterable[ResearchTask]) -> None:
        for task in tasks:
            if task.task_id in self._tasks:
                continue
            self._tasks[task.task_id] = task
            self._append_event("research_task_registered", task.task_id, "supervisor")

    def ingest(self, contribution: AgentContribution, run_record: Optional[AgentRunRecord] = None) -> int:
        if contribution.contribution_id in self._contributions:
            return self.version

        aliases: Dict[str, str] = {}
        for source in contribution.sources:
            canonical_key = self._canonical_source_key(source)
            canonical_source_id = stable_id("src", canonical_key)
            aliases[source.source_id] = canonical_source_id
            self._source_aliases[source.source_id] = canonical_source_id
            if canonical_source_id not in self._sources:
                self._sources[canonical_source_id] = EvidenceCandidate(
                    source_id=canonical_source_id,
                    platform=source.platform,
                    source_type=source.source_type,
                    source_name=source.source_name,
                    url=source.url,
                    canonical_url=canonicalize_url(source.canonical_url or source.url),
                    title=source.title,
                    text=source.text,
                    language=source.language,
                    source_item_id=source.source_item_id,
                    author_id_hash=source.author_id_hash,
                    published_at=source.published_at,
                    metadata=dict(source.metadata),
                )
                self._append_event("source_added", canonical_source_id, contribution.contribution_id)
            else:
                current = self._sources[canonical_source_id]
                if len(source.text) > len(current.text) or (not current.published_at and source.published_at):
                    self._sources[canonical_source_id] = EvidenceCandidate(
                        source_id=canonical_source_id,
                        platform=current.platform or source.platform,
                        source_type=current.source_type if current.source_type != "search_result" else source.source_type,
                        source_name=current.source_name or source.source_name,
                        url=current.url or source.url,
                        canonical_url=current.canonical_url or canonicalize_url(source.canonical_url or source.url),
                        title=source.title if len(source.title) > len(current.title) else current.title,
                        text=source.text if len(source.text) > len(current.text) else current.text,
                        language=current.language if current.language != "unknown" else source.language,
                        source_item_id=current.source_item_id or source.source_item_id,
                        author_id_hash=current.author_id_hash or source.author_id_hash,
                        published_at=current.published_at or source.published_at,
                        metadata={**current.metadata, **source.metadata},
                    )
                    self._append_event("source_enriched", canonical_source_id, contribution.contribution_id)

        for observation in contribution.acquisitions:
            source_id = aliases.get(observation.source_id) or self._source_aliases.get(observation.source_id, observation.source_id)
            observation_id = stable_id(
                "obs",
                contribution.contribution_id,
                observation.task_id,
                observation.agent,
                observation.query,
                observation.provider,
                observation.tool,
                source_id,
                str(observation.rank),
            )
            if observation_id not in self._acquisitions:
                self._acquisitions[observation_id] = AcquisitionObservation(
                    observation_id=observation_id,
                    source_id=source_id,
                    task_id=observation.task_id,
                    agent=observation.agent,
                    query=observation.query,
                    provider=observation.provider,
                    tool=observation.tool,
                    observed_at=observation.observed_at,
                    retrieved_at=observation.retrieved_at,
                    rank=observation.rank,
                    score=observation.score,
                    raw_ref=observation.raw_ref,
                    metadata=dict(observation.metadata),
                )
                self._append_event("acquisition_observed", observation_id, contribution.contribution_id)

        span_aliases: Dict[str, str] = {}
        for span in contribution.evidence_spans:
            source_id = aliases.get(span.source_id) or self._source_aliases.get(span.source_id, span.source_id)
            span_id = stable_id("esp", source_id, span.modality, str(span.start_char), str(span.end_char), span.text)
            span_aliases[span.span_id] = span_id
            if span_id not in self._spans:
                self._spans[span_id] = EvidenceSpan(
                    span_id=span_id,
                    source_id=source_id,
                    text=span.text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    span_type=span.span_type,
                    modality=span.modality,
                    locator=dict(span.locator),
                    extraction_route=span.extraction_route,
                    confidence=span.confidence,
                )
                self._append_event("evidence_span_added", span_id, contribution.contribution_id)

        for proposal in contribution.claim_proposals:
            proposal_id = stable_id("cp", proposal.agent, proposal.claim_text, proposal.task_id)
            if proposal_id not in self._proposals:
                self._proposals[proposal_id] = ClaimProposal(
                    proposal_id=proposal_id,
                    agent=proposal.agent,
                    claim_text=proposal.claim_text,
                    claim_type=proposal.claim_type,
                    target_entity=proposal.target_entity,
                    aspect=proposal.aspect,
                    stance=proposal.stance,
                    evidence_span_ids=[span_aliases.get(item, item) for item in proposal.evidence_span_ids],
                    task_id=proposal.task_id,
                    confidence=proposal.confidence,
                    uncertainty=list(proposal.uncertainty),
                )
                self._append_event("claim_proposed", proposal_id, contribution.contribution_id)

        if contribution.coverage:
            self._coverage[contribution.coverage.assessment_id] = contribution.coverage
            self._append_event("coverage_assessed", contribution.coverage.assessment_id, contribution.contribution_id)

        if isinstance(contribution, MediaContribution):
            for dossier in contribution.dossiers:
                normalized = SectionDossier(
                    dossier_id=dossier.dossier_id,
                    task_id=dossier.task_id,
                    section_id=dossier.section_id,
                    title=dossier.title,
                    objective=dossier.objective,
                    summary=dossier.summary,
                    source_ids=[aliases.get(item) or self._source_aliases.get(item, item) for item in dossier.source_ids],
                    evidence_span_ids=[span_aliases.get(item, item) for item in dossier.evidence_span_ids],
                    multimodal_assets=list(dossier.multimodal_assets),
                    unresolved_questions=list(dossier.unresolved_questions),
                    reflection_rounds=dossier.reflection_rounds,
                    status=dossier.status,
                )
                self._dossiers[normalized.dossier_id] = normalized
                self._append_event("section_dossier_added", normalized.dossier_id, contribution.contribution_id)

        if run_record:
            self._agent_runs[run_record.run_id] = run_record
            self._append_event("agent_run_recorded", run_record.run_id, contribution.contribution_id)
        self._contributions.add(contribution.contribution_id)
        self._append_event("contribution_ingested", contribution.contribution_id, contribution.contribution_id)
        return self.version

    def record_agent_run(self, run_record: AgentRunRecord) -> None:
        if run_record.run_id in self._agent_runs:
            return
        self._agent_runs[run_record.run_id] = run_record
        self._append_event("agent_run_recorded", run_record.run_id, "supervisor")

    def normalized_items(self) -> List[NormalizedItem]:
        observations_by_source: Dict[str, List[AcquisitionObservation]] = {}
        for observation in self._acquisitions.values():
            observations_by_source.setdefault(observation.source_id, []).append(observation)

        rows: List[NormalizedItem] = []
        for source_id, source in self._sources.items():
            observations = sorted(observations_by_source.get(source_id, []), key=lambda item: item.retrieved_at)
            first = observations[0] if observations else None
            rows.append(
                NormalizedItem(
                    item_id=source_id,
                    raw_id=source.source_item_id or source_id,
                    platform=source.platform or "web",
                    source_type=source.source_type or "search_result",
                    source_name=source.source_name or source.platform or "unknown",
                    source_item_id=source.source_item_id or source_id,
                    url=source.url,
                    canonical_url=source.canonical_url or canonicalize_url(source.url),
                    author_id_hash=source.author_id_hash,
                    title=source.title,
                    text=source.text[:4000],
                    language=source.language,
                    published_at=source.published_at,
                    observed_at=first.observed_at if first else utc_now(),
                    retrieved_at=first.retrieved_at if first else utc_now(),
                    # Compatibility only. Complete discovery provenance is in acquisition_observations.
                    retrieval_query=first.query if first else "",
                    raw_ref=f"blackboard://{self.run_id}/{source_id}",
                    normalization_version="fusion_norm_v1",
                    acquisition_source=first.provider if first else "specialist",
                )
            )
        return rows

    def snapshot(self) -> EvidenceBlackboardSnapshot:
        return EvidenceBlackboardSnapshot(
            version=self.version,
            sources=list(self._sources.values()),
            acquisitions=list(self._acquisitions.values()),
            evidence_spans=list(self._spans.values()),
            claim_proposals=list(self._proposals.values()),
            coverage_assessments=list(self._coverage.values()),
            section_dossiers=list(self._dossiers.values()),
            research_tasks=list(self._tasks.values()),
            agent_runs=list(self._agent_runs.values()),
            events=list(self._events),
        )

    def _append_event(self, event_type: str, entity_id: str, contribution_id: str) -> None:
        self.version += 1
        self._events.append(
            BlackboardEvent(
                version=self.version,
                event_type=event_type,
                entity_id=entity_id,
                contribution_id=contribution_id,
                recorded_at=utc_now(),
            )
        )

    @staticmethod
    def _canonical_source_key(source: EvidenceCandidate) -> str:
        canonical_url = canonicalize_url(source.canonical_url or source.url)
        if canonical_url:
            return f"url:{canonical_url}"
        content = " ".join(f"{source.title} {source.text}".lower().split())[:1000]
        return f"content:{content}"
