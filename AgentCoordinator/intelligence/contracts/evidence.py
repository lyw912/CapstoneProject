"""Evidence-first contracts for the AgentCoordinator intelligence layer.

These dataclasses are intentionally plain Python objects so the runtime can run
without a database or heavy schema dependency. The contract boundary is still
strict: final insights must link back to claims, claims must link to source
spans, and source spans must link to normalized items.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "coordinator_intelligence_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


@dataclass
class ContractMixin:
    def to_dict(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass
class ProviderDiagnostic(ContractMixin):
    provider: str
    capability: str
    status: str
    route: str
    configured: bool
    required: bool = False
    model: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchTraceStep(ContractMixin):
    node: str
    route: str
    input_count: int
    output_count: int
    elapsed_ms: int
    schema_version: str = SCHEMA_VERSION
    model: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RetrievalTask(ContractMixin):
    task_id: str
    parent_claim_id: Optional[str]
    query: str
    query_variants: List[str]
    target_source: str
    purpose: str
    priority: int
    deadline_sec: int
    max_results: int
    budget: Dict[str, Any]
    created_by: str


@dataclass
class RetrievalTaskResult(ContractMixin):
    task_id: str
    provider: str
    status: str
    items_returned: int
    errors: List[str] = field(default_factory=list)
    elapsed_ms: int = 0


@dataclass
class RunBudget(ContractMixin):
    """Supervisor ceilings; deadline/source caps are enforced at the boundary."""

    max_rounds: int = 1
    max_tasks: int = 8
    max_sources: int = 80
    max_api_calls: int = 20
    max_llm_calls: int = 20
    deadline_sec: int = 180


@dataclass
class ResearchTask(ContractMixin):
    """Typed delegation unit shared by the parent and specialist subgraphs."""

    task_id: str
    agent: str
    objective: str
    query: str
    task_type: str
    output_contract: str
    priority: int = 3
    round_index: int = 0
    target_claim_id: Optional[str] = None
    section_id: Optional[str] = None
    required_stances: List[str] = field(default_factory=list)
    source_scope: List[str] = field(default_factory=list)
    budget: RunBudget = field(default_factory=RunBudget)
    created_by: str = "fusion_supervisor"


@dataclass
class EvidenceCandidate(ContractMixin):
    """Source-shaped specialist output before canonical EvidenceCore ingest."""

    source_id: str
    platform: str
    source_type: str
    source_name: str
    url: str
    canonical_url: str
    title: str
    text: str
    language: str = "unknown"
    source_item_id: str = ""
    author_id_hash: Optional[str] = None
    published_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionObservation(ContractMixin):
    """One discovery of a source; never collapsed into the source entity."""

    observation_id: str
    source_id: str
    task_id: str
    agent: str
    query: str
    provider: str
    tool: str
    observed_at: str
    retrieved_at: str
    rank: Optional[int] = None
    score: Optional[float] = None
    raw_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSpan(ContractMixin):
    """Addressable excerpt proposed by a specialist before/after normalization."""

    span_id: str
    source_id: str
    text: str
    start_char: int
    end_char: int
    span_type: str
    modality: str = "text"
    locator: Dict[str, Any] = field(default_factory=dict)
    extraction_route: str = "specialist"
    confidence: float = 0.0


@dataclass
class ClaimProposal(ContractMixin):
    proposal_id: str
    agent: str
    claim_text: str
    claim_type: str
    target_entity: str
    aspect: str
    stance: str
    evidence_span_ids: List[str]
    task_id: str
    confidence: float
    uncertainty: List[str] = field(default_factory=list)


@dataclass
class EvidenceRelationEdge(ContractMixin):
    edge_id: str
    relation: str
    from_id: str
    to_id: str
    evidence_span_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class CoverageAssessment(ContractMixin):
    assessment_id: str
    task_id: str
    agent: str
    score: float
    stance_counts: Dict[str, int] = field(default_factory=dict)
    source_type_counts: Dict[str, int] = field(default_factory=dict)
    covered_dimensions: List[str] = field(default_factory=list)
    missing_dimensions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


@dataclass
class SectionDossier(ContractMixin):
    dossier_id: str
    task_id: str
    section_id: str
    title: str
    objective: str
    summary: str
    source_ids: List[str]
    evidence_span_ids: List[str]
    multimodal_assets: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    reflection_rounds: int = 0
    status: str = "complete"


@dataclass
class AgentRunRecord(ContractMixin):
    run_id: str
    task_id: str
    agent: str
    status: str
    started_at: str
    completed_at: str
    elapsed_ms: int
    source_count: int
    llm_calls: Optional[int] = None
    api_calls: Optional[int] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class AgentContribution(ContractMixin):
    contribution_id: str
    task_id: str
    agent: str
    status: str
    sources: List[EvidenceCandidate] = field(default_factory=list)
    acquisitions: List[AcquisitionObservation] = field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    claim_proposals: List[ClaimProposal] = field(default_factory=list)
    coverage: Optional[CoverageAssessment] = None
    trace: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class QueryContribution(AgentContribution):
    stance_distribution: Dict[str, float] = field(default_factory=dict)
    opinion_clusters: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    social_sentiment: Optional[Dict[str, Any]] = None


@dataclass
class MediaContribution(AgentContribution):
    dossiers: List[SectionDossier] = field(default_factory=list)
    narrative_summary: str = ""


@dataclass
class NormalizedItem(ContractMixin):
    item_id: str
    raw_id: str
    platform: str
    source_type: str
    source_name: str
    source_item_id: str
    url: str
    canonical_url: str
    author_id_hash: Optional[str]
    title: str
    text: str
    language: str
    published_at: Optional[str]
    observed_at: str
    retrieved_at: str
    retrieval_query: str
    raw_ref: str
    normalization_version: str = "norm_v1"
    acquisition_source: str = "unknown"


@dataclass
class QualityFeatures(ContractMixin):
    item_id: str
    canonical_item_id: str
    dup_type: str
    dup_confidence: float
    amplification_count: int
    copy_ratio_in_cluster: float
    relevance_score: float
    informativeness_score: float
    originality_score: float
    source_authority_score: float
    freshness_score: float
    stance: str
    stance_confidence: float
    sentiment: str
    sentiment_confidence: float
    aspect: str
    coordination_score: float
    persuasiveness_score: float
    low_quality_reasons: List[str]
    judge_route: str
    feature_version: str = "qf_v1"


@dataclass
class CanonicalCluster(ContractMixin):
    canonical_item_id: str
    representative_item_id: str
    member_item_ids: List[str]
    cluster_type: str
    amplification_count: int
    unique_author_count: int
    platforms: List[str]
    first_seen_at: str
    last_seen_at: str
    representative_reason: str


@dataclass
class SourceSpan(ContractMixin):
    span_id: str
    evidence_id: str
    text: str
    start_char: int
    end_char: int
    span_type: str
    extraction_route: str
    confidence: float


@dataclass
class EvidenceItem(ContractMixin):
    evidence_id: str
    item_id: str
    canonical_item_id: str
    source_type: str
    platform: str
    title: str
    text: str
    url: str
    source_name: str
    published_at: Optional[str]
    quality_ref: str
    spans: List[SourceSpan]
    acquisition_source: str = "unknown"


@dataclass
class Claim(ContractMixin):
    claim_id: str
    claim_text: str
    claim_type: str
    target_entity: str
    aspect: str
    time_scope: str
    stance: str
    sentiment: str
    supporting_spans: List[str]
    contradicting_spans: List[str]
    quality_summary: Dict[str, Any]
    status: str
    confidence: float
    created_by: str
    model: str
    schema_version: str = "claim_v1"


@dataclass
class ContradictionEdge(ContractMixin):
    edge_id: str
    claim_a: str
    claim_b: str
    relation: str
    explanation: str
    severity: str
    requires_follow_up: bool


@dataclass
class AuditDecision(ContractMixin):
    decision_id: str
    claim_id: str
    auditor: str
    decision: str
    reason_codes: List[str]
    explanation: str
    required_edit: str
    follow_up_tasks: List[str]
    confidence: float


@dataclass
class FreshnessSummary(ContractMixin):
    newest_published_at: Optional[str]
    oldest_published_at: Optional[str]
    median_age_hours: Optional[float]
    retrieval_lag_p95_sec: Optional[float]
    stale_source_ratio: float


@dataclass
class Insight(ContractMixin):
    insight_id: str
    title: str
    body: str
    claim_ids: List[str]
    citation_spans: List[str]
    counter_evidence_spans: List[str]
    strength: str
    wording_policy: str
    quality_warnings: List[str]
    freshness: Dict[str, Any]


@dataclass
class EvidenceGraph(ContractMixin):
    normalized_items: List[NormalizedItem] = field(default_factory=list)
    quality_features: List[QualityFeatures] = field(default_factory=list)
    canonical_clusters: List[CanonicalCluster] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    contradiction_edges: List[ContradictionEdge] = field(default_factory=list)
    retrieval_tasks: List[RetrievalTask] = field(default_factory=list)
    retrieval_results: List[RetrievalTaskResult] = field(default_factory=list)
    audit_decisions: List[AuditDecision] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    acquisition_observations: List[AcquisitionObservation] = field(default_factory=list)
    proposed_evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    claim_proposals: List[ClaimProposal] = field(default_factory=list)
    support_edges: List[EvidenceRelationEdge] = field(default_factory=list)
    coverage_assessments: List[CoverageAssessment] = field(default_factory=list)
    section_dossiers: List[SectionDossier] = field(default_factory=list)
    research_tasks: List[ResearchTask] = field(default_factory=list)
    agent_runs: List[AgentRunRecord] = field(default_factory=list)
    blackboard_version: int = 0

    def item_index(self) -> Dict[str, NormalizedItem]:
        return {item.item_id: item for item in self.normalized_items}

    def quality_index(self) -> Dict[str, QualityFeatures]:
        return {item.item_id: item for item in self.quality_features}

    def span_index(self) -> Dict[str, SourceSpan]:
        return {
            span.span_id: span
            for evidence in self.evidence_items
            for span in evidence.spans
        }

    def claim_index(self) -> Dict[str, Claim]:
        return {claim.claim_id: claim for claim in self.claims}

    def graph_summary(self) -> Dict[str, Any]:
        supported = sum(1 for claim in self.claims if claim.status == "supported")
        disputed = sum(1 for claim in self.claims if claim.status == "disputed")
        unsupported = sum(1 for claim in self.claims if claim.status in {"unsupported", "demoted"})
        raw_count = len(self.normalized_items)
        canonical_count = len(self.canonical_clusters)
        duplicate_count = max(0, raw_count - canonical_count)
        amplification_total = sum(max(0, cluster.amplification_count - 1) for cluster in self.canonical_clusters)
        return {
            "raw_count": raw_count,
            "normalized_count": len(self.normalized_items),
            "canonical_count": canonical_count,
            "claims_count": len(self.claims),
            "supported_claims": supported,
            "disputed_claims": disputed,
            "unsupported_removed": unsupported,
            "evidence_items": len(self.evidence_items),
            "source_spans": sum(len(item.spans) for item in self.evidence_items),
            "duplicate_count": duplicate_count,
            "amplification_count": amplification_total,
            "acquisition_observations": len(self.acquisition_observations),
            "section_dossiers": len(self.section_dossiers),
            "specialist_contributions": len(self.agent_runs),
            "blackboard_version": self.blackboard_version,
        }


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else default
