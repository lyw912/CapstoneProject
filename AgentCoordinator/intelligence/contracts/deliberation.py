"""Typed contracts for evidence-bound multi-agent deliberation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence import ContractMixin, utc_now


DELIBERATION_SCHEMA_VERSION = "evidence_debate_v1"


@dataclass
class InvestigationBrief(ContractMixin):
    original_query: str
    target_entity: str
    analysis_type: str
    factual_question: str
    discourse_question: str
    claim_modes: List[str]
    time_scope: str
    sample_boundary: str
    role_obligations: Dict[str, List[str]] = field(default_factory=dict)
    brief_version: str = "investigation_brief_v1"


@dataclass
class DebateAgentProfile(ContractMixin):
    role_id: str
    name: str
    chamber: str
    analytical_lens: str
    mandate: str
    evidence_obligations: List[str]
    prohibited_inferences: List[str]
    protocol_capabilities: List[str]
    model_route: str = "query"
    temperature: float = 0.2
    profile_version: str = "debate_profile_v1"


@dataclass
class EvidenceView(ContractMixin):
    view_id: str
    agent_id: str
    evidence_version: int
    shared_claim_ids: List[str]
    role_claim_ids: List[str]
    shared_span_ids: List[str]
    role_span_ids: List[str]
    evidence_item_ids: List[str]
    quality_warnings: List[str]
    selection_reasons: List[str]
    view_policy_version: str = "shared_core_role_slice_v1"

    @property
    def claim_ids(self) -> List[str]:
        return list(dict.fromkeys(self.shared_claim_ids + self.role_claim_ids))

    @property
    def span_ids(self) -> List[str]:
        return list(dict.fromkeys(self.shared_span_ids + self.role_span_ids))


@dataclass
class AgentPosition(ContractMixin):
    position_id: str
    agent_id: str
    claim_id: str
    stance: str
    argument: str
    evidence_span_ids: List[str]
    assumptions: List[str]
    uncertainties: List[str]
    confidence: float
    evidence_version: int
    round_index: int
    status: str = "sealed"
    created_at: str = field(default_factory=utc_now)


@dataclass
class MaterialClaimAssignment(ContractMixin):
    claim_id: str
    score: float
    reason_codes: List[str]
    assigned_reviewers: List[str] = field(default_factory=lambda: ["skeptic", "methodologist"])


@dataclass
class ArgumentAct(ContractMixin):
    act_id: str
    actor_id: str
    act_type: str
    target_claim_id: str
    content: str
    evidence_span_ids: List[str]
    reason_codes: List[str]
    evidence_version: int
    round_index: int
    target_position_id: Optional[str] = None
    target_act_id: Optional[str] = None
    requested_evidence: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=utc_now)


@dataclass
class PositionRevision(ContractMixin):
    revision_id: str
    agent_id: str
    claim_id: str
    previous_position_id: str
    triggering_act_ids: List[str]
    revision_type: str
    revised_argument: str
    revised_claim_text: Optional[str]
    evidence_span_ids: List[str]
    reason: str
    evidence_version: int
    round_index: int
    created_at: str = field(default_factory=utc_now)


@dataclass
class JudgeVerdict(ContractMixin):
    verdict_id: str
    judge_id: str
    claim_id: str
    decision: str
    reason_codes: List[str]
    explanation: str
    required_edit: str
    final_wording: Optional[str]
    decisive_act_ids: List[str]
    evidence_span_ids: List[str]
    confidence: float
    order_variant: str
    evidence_version: int
    created_at: str = field(default_factory=utc_now)


@dataclass
class ProtocolFailure(ContractMixin):
    failure_id: str
    phase: str
    agent_id: str
    failure_type: str
    message: str
    claim_id: Optional[str] = None
    retryable: bool = False
    created_at: str = field(default_factory=utc_now)


@dataclass
class DebateSession(ContractMixin):
    session_id: str
    run_id: str
    investigation_brief: InvestigationBrief
    profiles: List[DebateAgentProfile]
    evidence_views: List[EvidenceView] = field(default_factory=list)
    material_claims: List[MaterialClaimAssignment] = field(default_factory=list)
    positions: List[AgentPosition] = field(default_factory=list)
    argument_acts: List[ArgumentAct] = field(default_factory=list)
    revisions: List[PositionRevision] = field(default_factory=list)
    verdicts: List[JudgeVerdict] = field(default_factory=list)
    protocol_failures: List[ProtocolFailure] = field(default_factory=list)
    output_groups: Dict[str, List[str]] = field(default_factory=dict)
    independence_summary: Dict[str, Any] = field(default_factory=dict)
    budget_summary: Dict[str, Any] = field(default_factory=dict)
    status: str = "planned"
    schema_version: str = DELIBERATION_SCHEMA_VERSION
