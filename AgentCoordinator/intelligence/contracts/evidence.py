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
        }


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else default
