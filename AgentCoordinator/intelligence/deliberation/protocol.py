"""Bounded dual-chamber protocol over a versioned EvidenceGraph."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..contracts import (
    AgentPosition,
    ArgumentAct,
    AuditDecision,
    DebateAgentProfile,
    DebateSession,
    EvidenceGraph,
    EvidenceView,
    InvestigationBrief,
    JudgeVerdict,
    MaterialClaimAssignment,
    PositionRevision,
)
from .ledger import ArgumentLedger, InvalidArgumentReference
from .profiles import build_role_profiles
from .runner import DebateRunner, OpenAICompatibleDebateRunner


POSITION_STANCES = {"support", "challenge", "qualify", "abstain"}
REVIEW_ACTS = {"challenge", "qualify", "request_evidence", "support"}
RESPONSE_ACTS = {"rebut", "revise", "concede", "abstain", "request_evidence"}
VERDICTS = {"accept", "weaken", "reject", "needs_search", "unresolved"}


class DualChamberDeliberation:
    """Execute real isolated LLM roles while EvidenceCore retains evidence ownership."""

    def __init__(
        self,
        run_id: str,
        brief: InvestigationBrief,
        settings: Any = None,
        runner: Optional[DebateRunner] = None,
    ):
        self.settings = settings
        self.enabled = bool(getattr(settings, "COORDINATOR_ENABLE_DEBATE", False))
        self.max_material_claims = max(1, int(getattr(settings, "COORDINATOR_DEBATE_MAX_MATERIAL_CLAIMS", 6)))
        self.max_calls = max(1, int(getattr(settings, "COORDINATOR_DEBATE_MAX_LLM_CALLS", 18)))
        self.deadline_sec = max(1, int(getattr(settings, "COORDINATOR_DEBATE_TIMEOUT", 600)))
        self.schema_retries = max(0, int(getattr(settings, "COORDINATOR_DEBATE_SCHEMA_RETRIES", 1)))
        self.started = time.monotonic()
        profiles = build_role_profiles(brief.analysis_type, settings=settings)
        self.session = DebateSession(
            session_id=f"debate_{run_id}",
            run_id=run_id,
            investigation_brief=brief,
            profiles=profiles,
            independence_summary={
                "context_isolated": True,
                "objective_distinct": True,
                "model_family_distinct": False,
                "configured_mode": "same_model_fallback",
            },
            budget_summary={
                "max_material_claims": self.max_material_claims,
                "max_llm_calls": self.max_calls,
                "deadline_sec": self.deadline_sec,
                "llm_calls": 0,
                "calls_by_phase": {},
                "termination_reason": "not_started",
            },
        )
        self.ledger = ArgumentLedger(self.session)
        self.runner = runner or (OpenAICompatibleDebateRunner(settings) if settings is not None else None)
        self._profiles = {profile.role_id: profile for profile in profiles}
        self._models_used: Dict[str, str] = {}

    async def open_or_reassess(
        self,
        graph: EvidenceGraph,
        quality_warnings: Optional[List[str]] = None,
        affected_claim_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.enabled:
            self.session.status = "disabled"
            self._finish_budget("disabled")
            return
        if not graph.claims:
            self.ledger.failure("perspective_opening", "coordinator", "no_claims", "No EvidenceCore claims were available for deliberation.")
            self.session.status = "partial"
            return
        if affected_claim_ids and self.session.positions:
            await self._reassess(graph, list(dict.fromkeys(affected_claim_ids)), quality_warnings or [])
            return

        self.session.status = "perspective_opening"
        perspective_profiles = [item for item in self.session.profiles if item.chamber == "perspective"]
        views = [self._build_view(graph, profile, quality_warnings or []) for profile in perspective_profiles]
        self.session.evidence_views.extend(views)
        await asyncio.gather(
            *(self._opening_call(profile, view, graph) for profile, view in zip(perspective_profiles, views))
        )
        for position in self.session.positions:
            if position.status == "sealed":
                position.status = "published"
        self._select_material_claims(graph)
        self.session.status = "openings_published"

    async def review_material_claims(self, graph: EvidenceGraph) -> None:
        if not self.enabled or not self.session.material_claims:
            return
        if any(item.actor_id in {"skeptic", "methodologist"} for item in self.session.argument_acts):
            return
        self.session.status = "evidence_review"
        material_ids = [item.claim_id for item in self.session.material_claims]
        await asyncio.gather(
            self._review_call(self._profiles["skeptic"], graph, material_ids),
            self._review_call(self._profiles["methodologist"], graph, material_ids),
        )
        self.session.status = "review_complete"

    async def collect_proposer_responses(self, graph: EvidenceGraph) -> None:
        if not self.enabled:
            return
        responded_challenges = {
            item.target_act_id
            for item in self.session.argument_acts
            if item.target_act_id and item.actor_id not in {"skeptic", "methodologist"}
        }
        challenges = [
            item
            for item in self.session.argument_acts
            if item.actor_id in {"skeptic", "methodologist"} and item.act_type in {"challenge", "qualify", "request_evidence"}
            and item.act_id not in responded_challenges
        ]
        if not challenges:
            return
        by_agent: Dict[str, List[ArgumentAct]] = defaultdict(list)
        positions_by_id = {item.position_id: item for item in self.session.positions}
        for challenge in challenges:
            if challenge.target_position_id and challenge.target_position_id in positions_by_id:
                by_agent[positions_by_id[challenge.target_position_id].agent_id].append(challenge)
                continue
            for position in self.session.positions:
                if position.claim_id == challenge.target_claim_id:
                    by_agent[position.agent_id].append(challenge)
        self.session.status = "proposer_response"
        await asyncio.gather(
            *(
                self._response_call(self._profiles[agent_id], graph, list(dict.fromkeys(item.act_id for item in acts)))
                for agent_id, acts in by_agent.items()
                if agent_id in self._profiles
            )
        )
        self.session.status = "response_complete"

    def requested_evidence_claim_ids(self) -> List[str]:
        return list(
            dict.fromkeys(
                item.target_claim_id
                for item in self.session.argument_acts
                if item.act_type == "request_evidence"
            )
        )

    async def adjudicate(
        self,
        graph: EvidenceGraph,
        deterministic_decisions: Sequence[AuditDecision],
    ) -> List[AuditDecision]:
        if not self.enabled or not self.session.material_claims:
            decisions = list(deterministic_decisions)
            self._group_outputs(graph, decisions)
            self.session.status = "disabled" if not self.enabled else "complete_without_material_review"
            self._finish_budget(self.session.status)
            return decisions

        material_ids = [item.claim_id for item in self.session.material_claims]
        self.session.status = "paired_adjudication"
        primary, review = await asyncio.gather(
            self._judge_call(self._profiles["primary_judge"], graph, material_ids, "original"),
            self._judge_call(self._profiles["review_judge"], graph, material_ids, "reversed"),
        )
        primary_by_claim = {item.claim_id: item for item in primary}
        review_by_claim = {item.claim_id: item for item in review}
        deterministic = {item.claim_id: item for item in deterministic_decisions}
        merged: List[AuditDecision] = []

        for claim in graph.claims:
            base = deterministic.get(claim.claim_id)
            if base is None:
                continue
            if claim.claim_id not in material_ids or base.decision == "reject":
                merged.append(base)
                self._apply_claim_status(claim, base.decision, None, base.confidence)
                continue
            left = primary_by_claim.get(claim.claim_id)
            right = review_by_claim.get(claim.claim_id)
            if left is None or right is None:
                reason_codes = list(dict.fromkeys(base.reason_codes + ["paired_judge_incomplete"]))
                fallback = AuditDecision(
                    decision_id=f"ad_debate_{claim.claim_id}",
                    claim_id=claim.claim_id,
                    auditor="deterministic_gate_with_incomplete_pair",
                    decision=("weaken" if base.decision == "accept" else base.decision),
                    reason_codes=reason_codes,
                    explanation="Paired adjudication was incomplete; deterministic evidence eligibility remains authoritative.",
                    required_edit="Use cautious wording and retain the paired-judge failure diagnostic.",
                    follow_up_tasks=base.follow_up_tasks,
                    confidence=min(base.confidence, 0.55),
                )
                merged.append(fallback)
                self._apply_claim_status(claim, fallback.decision, None, fallback.confidence)
                continue

            decision = self._merge_verdicts(left.decision, right.decision)
            reasons = list(dict.fromkeys(base.reason_codes + left.reason_codes + right.reason_codes))
            if left.decision != right.decision:
                reasons.append("paired_judge_disagreement")
            final_wording = self._final_wording(decision, left, right)
            required_edit = right.required_edit or left.required_edit or base.required_edit
            merged_decision = AuditDecision(
                decision_id=f"ad_debate_{claim.claim_id}",
                claim_id=claim.claim_id,
                auditor="primary_and_review_judge",
                decision=decision,
                reason_codes=list(dict.fromkeys(reasons)),
                explanation=(
                    f"Primary judge: {left.explanation} Review judge: {right.explanation}"
                    if left.decision == right.decision
                    else "The isolated judges disagreed, so the protocol did not force a winning verdict."
                ),
                required_edit=required_edit,
                follow_up_tasks=base.follow_up_tasks,
                confidence=round(min(left.confidence, right.confidence, base.confidence), 4),
            )
            merged.append(merged_decision)
            self._apply_claim_status(claim, decision, final_wording, merged_decision.confidence)

        graph.audit_decisions = merged
        self._group_outputs(graph, merged)
        self.session.status = "complete" if not self.session.protocol_failures else "complete_with_diagnostics"
        self._finish_budget("paired_adjudication_complete")
        return merged

    def diagnostics(self) -> List[str]:
        return [f"{item.phase}:{item.agent_id}:{item.failure_type}: {item.message}" for item in self.session.protocol_failures]

    async def _opening_call(self, profile: DebateAgentProfile, view: EvidenceView, graph: EvidenceGraph) -> None:
        payload = {
            "phase": "sealed_opening",
            "investigation_brief": self.session.investigation_brief.to_dict(),
            "agent_profile": profile.to_dict(),
            "evidence_view": view.to_dict(),
            "evidence": self._render_evidence(graph, view.claim_ids, view.span_ids),
            "output_schema": {
                "positions": [
                    {
                        "claim_id": "existing claim ID",
                        "stance": "support|challenge|qualify|abstain",
                        "argument": "evidence-bound analysis",
                        "evidence_span_ids": ["existing span ID"],
                        "assumptions": [],
                        "uncertainties": [],
                        "confidence": 0.0,
                    }
                ]
            },
        }
        response = await self._invoke(profile, "sealed_opening", self._opening_system_prompt(profile), payload)
        rows = response.get("positions") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            self.ledger.failure("sealed_opening", profile.role_id, "invalid_envelope", "Missing positions array after schema retries.")
            return
        accepted = 0
        for index, row in enumerate(rows[: self.max_material_claims], start=1):
            if not isinstance(row, dict):
                continue
            try:
                stance = str(row.get("stance") or "").lower()
                if stance not in POSITION_STANCES:
                    raise InvalidArgumentReference(f"Unsupported stance: {stance}")
                position = AgentPosition(
                    position_id=self._id("pos", profile.role_id, str(row.get("claim_id")), str(graph.blackboard_version), str(index)),
                    agent_id=profile.role_id,
                    claim_id=str(row.get("claim_id") or ""),
                    stance=stance,
                    argument=str(row.get("argument") or "").strip(),
                    evidence_span_ids=self._strings(row.get("evidence_span_ids")),
                    assumptions=self._strings(row.get("assumptions")),
                    uncertainties=self._strings(row.get("uncertainties")),
                    confidence=self._confidence(row.get("confidence")),
                    evidence_version=graph.blackboard_version,
                    round_index=0,
                )
                self.ledger.add_position(position, graph, view.claim_ids)
                accepted += 1
            except (InvalidArgumentReference, TypeError, ValueError) as exc:
                self.ledger.failure("sealed_opening", profile.role_id, "invalid_position", str(exc))
        if not accepted:
            self.ledger.failure("sealed_opening", profile.role_id, "empty_valid_output", "No valid evidence-bound position was admitted.")

    async def _review_call(self, profile: DebateAgentProfile, graph: EvidenceGraph, material_ids: List[str]) -> None:
        payload = {
            "phase": "evidence_review",
            "investigation_brief": self.session.investigation_brief.to_dict(),
            "reviewer": profile.to_dict(),
            "material_claims": self._claim_subgraphs(graph, material_ids, include_actor_ids=True),
            "output_schema": {
                "acts": [
                    {
                        "act_type": "challenge|qualify|request_evidence|support",
                        "target_claim_id": "claim ID",
                        "target_position_id": "optional position ID",
                        "content": "review finding",
                        "evidence_span_ids": ["span ID"],
                        "reason_codes": ["single_source|missing_evidence|sample_boundary|alternative_explanation"],
                        "requested_evidence": {"purpose": "support|refute|clarify", "source_type": "optional"},
                    }
                ]
            },
        }
        response = await self._invoke(profile, "evidence_review", self._review_system_prompt(profile), payload)
        rows = response.get("acts") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            self.ledger.failure("evidence_review", profile.role_id, "invalid_envelope", "Missing acts array after schema retries.")
            return
        for index, row in enumerate(rows[: self.max_material_claims * 2], start=1):
            if not isinstance(row, dict):
                continue
            try:
                act_type = str(row.get("act_type") or "").lower()
                if act_type not in REVIEW_ACTS:
                    raise InvalidArgumentReference(f"Unsupported review act: {act_type}")
                act = ArgumentAct(
                    act_id=self._id("act", profile.role_id, str(row.get("target_claim_id")), str(index)),
                    actor_id=profile.role_id,
                    act_type=act_type,
                    target_claim_id=str(row.get("target_claim_id") or ""),
                    target_position_id=str(row.get("target_position_id") or "") or None,
                    content=str(row.get("content") or "").strip(),
                    evidence_span_ids=self._strings(row.get("evidence_span_ids")),
                    reason_codes=self._strings(row.get("reason_codes")),
                    requested_evidence=row.get("requested_evidence") if isinstance(row.get("requested_evidence"), dict) else None,
                    evidence_version=graph.blackboard_version,
                    round_index=1,
                )
                self.ledger.add_act(act, graph, material_ids)
            except (InvalidArgumentReference, TypeError, ValueError) as exc:
                self.ledger.failure("evidence_review", profile.role_id, "invalid_argument_act", str(exc))

    async def _response_call(
        self,
        profile: DebateAgentProfile,
        graph: EvidenceGraph,
        challenge_act_ids: List[str],
    ) -> None:
        challenge_index = {item.act_id: item for item in self.session.argument_acts}
        challenges = [challenge_index[item] for item in challenge_act_ids if item in challenge_index]
        claim_ids = list(dict.fromkeys(item.target_claim_id for item in challenges))
        own_positions = [item for item in self.session.positions if item.agent_id == profile.role_id and item.claim_id in claim_ids]
        if not own_positions:
            return
        payload = {
            "phase": "proposer_response",
            "investigation_brief": self.session.investigation_brief.to_dict(),
            "agent_profile": profile.to_dict(),
            "own_positions": [item.to_dict() for item in own_positions],
            "challenges": [item.to_dict() for item in challenges],
            "claim_subgraphs": self._claim_subgraphs(graph, claim_ids, include_actor_ids=False),
            "output_schema": {
                "responses": [
                    {
                        "act_type": "rebut|revise|concede|abstain|request_evidence",
                        "target_claim_id": "claim ID",
                        "target_act_id": "challenge act ID",
                        "content": "response",
                        "evidence_span_ids": ["span ID"],
                        "reason_codes": [],
                        "revised_claim_text": "required for revise when wording changes",
                        "requested_evidence": {"purpose": "support|refute|clarify"},
                    }
                ]
            },
        }
        response = await self._invoke(profile, "proposer_response", self._response_system_prompt(profile), payload)
        rows = response.get("responses") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            self.ledger.failure("proposer_response", profile.role_id, "invalid_envelope", "Missing responses array after schema retries.")
            return
        positions_by_claim = {item.claim_id: item for item in own_positions}
        allowed_trigger_ids = set(challenge_act_ids)
        for index, row in enumerate(rows[: len(claim_ids) * 2], start=1):
            if not isinstance(row, dict):
                continue
            try:
                act_type = str(row.get("act_type") or "").lower()
                if act_type not in RESPONSE_ACTS:
                    raise InvalidArgumentReference(f"Unsupported response act: {act_type}")
                target_act_id = str(row.get("target_act_id") or "") or None
                if target_act_id and target_act_id not in allowed_trigger_ids:
                    raise InvalidArgumentReference(f"Response targets an unassigned challenge: {target_act_id}")
                claim_id = str(row.get("target_claim_id") or "")
                act = ArgumentAct(
                    act_id=self._id("act", profile.role_id, claim_id, "response", str(index)),
                    actor_id=profile.role_id,
                    act_type=act_type,
                    target_claim_id=claim_id,
                    target_position_id=positions_by_claim.get(claim_id).position_id if claim_id in positions_by_claim else None,
                    target_act_id=target_act_id,
                    content=str(row.get("content") or "").strip(),
                    evidence_span_ids=self._strings(row.get("evidence_span_ids")),
                    reason_codes=self._strings(row.get("reason_codes")),
                    requested_evidence=row.get("requested_evidence") if isinstance(row.get("requested_evidence"), dict) else None,
                    evidence_version=graph.blackboard_version,
                    round_index=2,
                )
                self.ledger.add_act(act, graph, claim_ids)
                if act_type in {"revise", "concede", "abstain"} and claim_id in positions_by_claim:
                    triggers = [target_act_id] if target_act_id else challenge_act_ids
                    revision = PositionRevision(
                        revision_id=self._id("rev", profile.role_id, claim_id, str(index)),
                        agent_id=profile.role_id,
                        claim_id=claim_id,
                        previous_position_id=positions_by_claim[claim_id].position_id,
                        triggering_act_ids=[item for item in triggers if item],
                        revision_type=act_type,
                        revised_argument=act.content,
                        revised_claim_text=str(row.get("revised_claim_text") or "").strip() or None,
                        evidence_span_ids=act.evidence_span_ids,
                        reason="; ".join(act.reason_codes) or "Response to evidence review",
                        evidence_version=graph.blackboard_version,
                        round_index=2,
                    )
                    self.ledger.add_revision(revision, graph)
            except (InvalidArgumentReference, TypeError, ValueError) as exc:
                self.ledger.failure("proposer_response", profile.role_id, "invalid_response", str(exc))

    async def _reassess(self, graph: EvidenceGraph, affected_claim_ids: List[str], quality_warnings: List[str]) -> None:
        material = {item.claim_id for item in self.session.material_claims}
        affected = [item for item in affected_claim_ids if item in material and item in graph.claim_index()]
        if not affected:
            return
        self.session.status = "post_retrieval_reassessment"
        by_agent: Dict[str, List[str]] = defaultdict(list)
        for position in self.session.positions:
            if position.claim_id in affected:
                by_agent[position.agent_id].append(position.claim_id)
        for agent_id, claim_ids in by_agent.items():
            profile = self._profiles.get(agent_id)
            if not profile:
                continue
            view = self._build_view(graph, profile, quality_warnings, forced_claim_ids=claim_ids)
            self.session.evidence_views.append(view)
            payload = {
                "phase": "post_retrieval_reassessment",
                "investigation_brief": self.session.investigation_brief.to_dict(),
                "agent_profile": profile.to_dict(),
                "prior_positions": [
                    item.to_dict()
                    for item in self.session.positions
                    if item.agent_id == agent_id and item.claim_id in claim_ids
                ],
                "new_evidence_view": view.to_dict(),
                "evidence": self._render_evidence(graph, claim_ids, view.span_ids),
                "output_schema": {
                    "responses": [
                        {
                            "act_type": "rebut|revise|concede|abstain",
                            "target_claim_id": "claim ID",
                            "content": "what changed after retrieval",
                            "evidence_span_ids": ["span ID"],
                            "reason_codes": [],
                            "revised_claim_text": "optional",
                        }
                    ]
                },
            }
            response = await self._invoke(profile, "post_retrieval_reassessment", self._response_system_prompt(profile), payload)
            rows = response.get("responses") if isinstance(response, dict) else None
            if not isinstance(rows, list):
                self.ledger.failure("post_retrieval_reassessment", agent_id, "invalid_envelope", "Missing responses array.")
                continue
            prior_by_claim = {
                item.claim_id: item
                for item in self.session.positions
                if item.agent_id == agent_id and item.claim_id in claim_ids
            }
            request_acts = [
                item.act_id
                for item in self.session.argument_acts
                if item.target_claim_id in claim_ids and item.act_type == "request_evidence"
            ]
            for index, row in enumerate(rows[: len(claim_ids)], start=1):
                if not isinstance(row, dict):
                    continue
                try:
                    claim_id = str(row.get("target_claim_id") or "")
                    act_type = str(row.get("act_type") or "").lower()
                    if act_type not in {"rebut", "revise", "concede", "abstain"}:
                        raise InvalidArgumentReference(f"Unsupported reassessment act: {act_type}")
                    act = ArgumentAct(
                        act_id=self._id("act", agent_id, claim_id, "reassessment", str(index)),
                        actor_id=agent_id,
                        act_type=act_type,
                        target_claim_id=claim_id,
                        target_position_id=prior_by_claim.get(claim_id).position_id if claim_id in prior_by_claim else None,
                        content=str(row.get("content") or "").strip(),
                        evidence_span_ids=self._strings(row.get("evidence_span_ids")),
                        reason_codes=self._strings(row.get("reason_codes")),
                        evidence_version=graph.blackboard_version,
                        round_index=3,
                    )
                    self.ledger.add_act(act, graph, claim_ids)
                    if act_type in {"revise", "concede", "abstain"} and claim_id in prior_by_claim:
                        revision = PositionRevision(
                            revision_id=self._id("rev", agent_id, claim_id, "reassessment", str(index)),
                            agent_id=agent_id,
                            claim_id=claim_id,
                            previous_position_id=prior_by_claim[claim_id].position_id,
                            triggering_act_ids=request_acts,
                            revision_type=act_type,
                            revised_argument=act.content,
                            revised_claim_text=str(row.get("revised_claim_text") or "").strip() or None,
                            evidence_span_ids=act.evidence_span_ids,
                            reason="Post-retrieval evidence reassessment",
                            evidence_version=graph.blackboard_version,
                            round_index=3,
                        )
                        self.ledger.add_revision(revision, graph)
                except (InvalidArgumentReference, TypeError, ValueError) as exc:
                    self.ledger.failure("post_retrieval_reassessment", agent_id, "invalid_reassessment", str(exc))
        self.session.status = "reassessment_complete"

    async def _judge_call(
        self,
        profile: DebateAgentProfile,
        graph: EvidenceGraph,
        material_ids: List[str],
        order_variant: str,
    ) -> List[JudgeVerdict]:
        subgraphs = self._claim_subgraphs(graph, material_ids, include_actor_ids=False)
        if order_variant == "reversed":
            for item in subgraphs:
                item["argument_acts"] = list(reversed(item["argument_acts"]))
                item["positions"] = list(reversed(item["positions"]))
        payload = {
            "phase": "paired_blind_adjudication",
            "investigation_brief": self.session.investigation_brief.to_dict(),
            "judge_profile": profile.to_dict(),
            "order_variant": order_variant,
            "claim_subgraphs": subgraphs,
            "rubric": {
                "empirical_fact": ["span entailment", "authority", "freshness", "counter-evidence"],
                "causal_or_forecast": ["alternatives", "scope", "uncertainty", "counter-evidence"],
                "discourse_observation": ["sample boundary", "deduplication", "source independence", "stance coverage"],
                "normative_interpretation": ["explicit value frame", "stakeholder scope", "evidence context"],
            },
            "output_schema": {
                "verdicts": [
                    {
                        "claim_id": "claim ID",
                        "decision": "accept|weaken|reject|needs_search|unresolved",
                        "reason_codes": [],
                        "explanation": "rubric-grounded reason",
                        "required_edit": "wording rule",
                        "final_wording": "optional audited wording",
                        "decisive_act_ids": ["existing act ID"],
                        "evidence_span_ids": ["existing span ID"],
                        "confidence": 0.0,
                    }
                ]
            },
        }
        response = await self._invoke(profile, "paired_blind_adjudication", self._judge_system_prompt(), payload)
        rows = response.get("verdicts") if isinstance(response, dict) else None
        if not isinstance(rows, list):
            self.ledger.failure("paired_blind_adjudication", profile.role_id, "invalid_envelope", "Missing verdicts array.")
            return []
        verdicts: List[JudgeVerdict] = []
        for index, row in enumerate(rows[: len(material_ids)], start=1):
            if not isinstance(row, dict):
                continue
            try:
                decision = str(row.get("decision") or "").lower()
                if decision not in VERDICTS:
                    raise InvalidArgumentReference(f"Unsupported verdict: {decision}")
                claim_id = str(row.get("claim_id") or "")
                verdict = JudgeVerdict(
                    verdict_id=self._id("verdict", profile.role_id, claim_id, order_variant, str(index)),
                    judge_id=profile.role_id,
                    claim_id=claim_id,
                    decision=decision,
                    reason_codes=self._strings(row.get("reason_codes")),
                    explanation=str(row.get("explanation") or "").strip(),
                    required_edit=str(row.get("required_edit") or "").strip(),
                    final_wording=str(row.get("final_wording") or "").strip() or None,
                    decisive_act_ids=self._strings(row.get("decisive_act_ids")),
                    evidence_span_ids=self._strings(row.get("evidence_span_ids")),
                    confidence=self._confidence(row.get("confidence")),
                    order_variant=order_variant,
                    evidence_version=graph.blackboard_version,
                )
                self.ledger.add_verdict(verdict, graph)
                verdicts.append(verdict)
            except (InvalidArgumentReference, TypeError, ValueError) as exc:
                self.ledger.failure("paired_blind_adjudication", profile.role_id, "invalid_verdict", str(exc))
        return verdicts

    async def _invoke(
        self,
        profile: DebateAgentProfile,
        phase: str,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.runner is None:
            self.ledger.failure(phase, profile.role_id, "provider_unavailable", "No debate runner is configured.")
            return {}
        for attempt in range(self.schema_retries + 1):
            if self._budget_exhausted():
                self.ledger.failure(phase, profile.role_id, "budget_exhausted", "Debate call cap or deadline reached.")
                self._finish_budget("budget_exhausted")
                return {}
            self._record_call(phase, profile)
            try:
                response = await self.runner.invoke(profile, phase, system_prompt, payload)
                if isinstance(response, dict):
                    return response
                raise ValueError("Runner returned a non-object envelope")
            except Exception as exc:
                if attempt >= self.schema_retries:
                    self.ledger.failure(phase, profile.role_id, "provider_or_schema_error", str(exc), retryable=False)
                    return {}
                payload = dict(payload)
                payload["repair_instruction"] = f"Previous output failed validation: {str(exc)[:300]}. Return only the required JSON object."
        return {}

    def _build_view(
        self,
        graph: EvidenceGraph,
        profile: DebateAgentProfile,
        quality_warnings: List[str],
        forced_claim_ids: Optional[List[str]] = None,
    ) -> EvidenceView:
        ranked = sorted(graph.claims, key=lambda item: (self._claim_priority(item), item.confidence), reverse=True)
        shared_claims = [item.claim_id for item in ranked[:3]]
        if forced_claim_ids:
            role_claims = [item for item in forced_claim_ids if item in graph.claim_index()]
        else:
            role_ranked = sorted(ranked, key=lambda item: self._role_relevance(profile, item), reverse=True)
            role_claims = [item.claim_id for item in role_ranked if item.claim_id not in shared_claims][:3]
        claim_index = graph.claim_index()
        shared_spans = self._claim_spans(claim_index, shared_claims)[:12]
        role_spans = [item for item in self._claim_spans(claim_index, role_claims) if item not in shared_spans][:12]
        span_index = graph.span_index()
        evidence_ids = list(
            dict.fromkeys(
                span_index[span_id].evidence_id
                for span_id in shared_spans + role_spans
                if span_id in span_index
            )
        )
        return EvidenceView(
            view_id=self._id("view", profile.role_id, str(graph.blackboard_version), ",".join(role_claims)),
            agent_id=profile.role_id,
            evidence_version=graph.blackboard_version,
            shared_claim_ids=shared_claims,
            role_claim_ids=role_claims,
            shared_span_ids=shared_spans,
            role_span_ids=role_spans,
            evidence_item_ids=evidence_ids,
            quality_warnings=list(quality_warnings),
            selection_reasons=[
                "Shared core contains highest-priority claims and global evidence warnings.",
                f"Role slice selected for obligations: {', '.join(profile.evidence_obligations)}.",
            ],
        )

    def _select_material_claims(self, graph: EvidenceGraph) -> None:
        assignments: List[MaterialClaimAssignment] = []
        positions_by_claim: Dict[str, List[AgentPosition]] = defaultdict(list)
        for position in self.session.positions:
            positions_by_claim[position.claim_id].append(position)
        for claim in graph.claims:
            score = 0.0
            reasons: List[str] = []
            if claim.status in {"disputed", "needs_search"}:
                score += 3.0
                reasons.append(claim.status)
            if claim.contradicting_spans:
                score += 2.0
                reasons.append("counter_evidence")
            diversity = float((claim.quality_summary or {}).get("source_diversity") or 0.0)
            if diversity < 0.5:
                score += 1.5
                reasons.append("single_source_or_low_diversity")
            copy_ratio = float((claim.quality_summary or {}).get("copy_ratio") or 0.0)
            if copy_ratio >= 0.35:
                score += 1.0
                reasons.append("amplification_or_copy_risk")
            if claim.claim_type in {"causal", "forecast", "risk", "stance", "opinion"}:
                score += 1.5
                reasons.append("high_inference_claim")
            stances = {item.stance for item in positions_by_claim.get(claim.claim_id, []) if item.stance != "abstain"}
            if len(stances) > 1:
                score += 2.0
                reasons.append("perspective_disagreement")
            assignments.append(MaterialClaimAssignment(claim_id=claim.claim_id, score=round(score, 3), reason_codes=reasons))
        assignments.sort(key=lambda item: (item.score, graph.claim_index()[item.claim_id].confidence), reverse=True)
        selected = [item for item in assignments if item.score > 0][: self.max_material_claims]
        if not selected:
            selected = assignments[: min(2, len(assignments))]
            for item in selected:
                item.reason_codes.append("coverage_sample")
        self.session.material_claims = selected

    def _render_evidence(self, graph: EvidenceGraph, claim_ids: Iterable[str], span_ids: Iterable[str]) -> Dict[str, Any]:
        claims = graph.claim_index()
        spans = graph.span_index()
        evidence_by_id = {item.evidence_id: item for item in graph.evidence_items}
        return {
            "claims": [claims[item].to_dict() for item in claim_ids if item in claims],
            "spans": [
                {
                    "span_id": span_id,
                    "text": spans[span_id].text[:900],
                    "span_type": spans[span_id].span_type,
                    "confidence": spans[span_id].confidence,
                    "evidence_id": spans[span_id].evidence_id,
                    "source": {
                        "title": evidence_by_id[spans[span_id].evidence_id].title,
                        "url": evidence_by_id[spans[span_id].evidence_id].url,
                        "platform": evidence_by_id[spans[span_id].evidence_id].platform,
                        "source_type": evidence_by_id[spans[span_id].evidence_id].source_type,
                    }
                    if spans[span_id].evidence_id in evidence_by_id
                    else {},
                }
                for span_id in span_ids
                if span_id in spans
            ],
        }

    def _claim_subgraphs(self, graph: EvidenceGraph, claim_ids: List[str], include_actor_ids: bool) -> List[Dict[str, Any]]:
        claim_index = graph.claim_index()
        span_index = graph.span_index()
        rows: List[Dict[str, Any]] = []
        for claim_id in claim_ids:
            claim = claim_index.get(claim_id)
            if not claim:
                continue
            spans = list(dict.fromkeys(claim.supporting_spans + claim.contradicting_spans))
            positions = [item.to_dict() for item in self.session.positions if item.claim_id == claim_id]
            acts = [item.to_dict() for item in self.session.argument_acts if item.target_claim_id == claim_id]
            revisions = [item.to_dict() for item in self.session.revisions if item.claim_id == claim_id]
            if not include_actor_ids:
                for index, position in enumerate(positions, start=1):
                    position["agent_id"] = f"anonymous_position_{index}"
                for index, act in enumerate(acts, start=1):
                    act["actor_id"] = f"anonymous_reviewer_{index}"
            rows.append(
                {
                    "claim": claim.to_dict(),
                    "positions": positions,
                    "argument_acts": acts,
                    "revisions": revisions,
                    "evidence_spans": [
                        {
                            "span_id": span_id,
                            "text": span_index[span_id].text[:900],
                            "span_type": span_index[span_id].span_type,
                            "confidence": span_index[span_id].confidence,
                        }
                        for span_id in spans
                        if span_id in span_index
                    ],
                }
            )
        return rows

    def _group_outputs(self, graph: EvidenceGraph, decisions: Sequence[AuditDecision]) -> None:
        claim_index = graph.claim_index()
        groups = {
            "audited_findings": [],
            "contested_findings": [],
            "perspective_tensions": [],
            "rejected_claims": [],
            "evidence_gaps": [],
        }
        for decision in decisions:
            claim = claim_index.get(decision.claim_id)
            if not claim:
                continue
            if decision.decision in {"accept", "weaken"}:
                groups["audited_findings"].append(claim.claim_id)
            elif decision.decision == "reject":
                groups["rejected_claims"].append(claim.claim_id)
            elif decision.decision == "needs_search":
                groups["evidence_gaps"].append(claim.claim_id)
            elif claim.claim_type in {"opinion", "normative", "value"}:
                groups["perspective_tensions"].append(claim.claim_id)
            else:
                groups["contested_findings"].append(claim.claim_id)
        self.session.output_groups = groups

    def _record_call(self, phase: str, profile: DebateAgentProfile) -> None:
        budget = self.session.budget_summary
        budget["llm_calls"] = int(budget.get("llm_calls") or 0) + 1
        by_phase = dict(budget.get("calls_by_phase") or {})
        by_phase[phase] = int(by_phase.get(phase) or 0) + 1
        budget["calls_by_phase"] = by_phase
        self._models_used[profile.role_id] = self.runner.model_name(profile) if self.runner else profile.model_route
        model_families = set(self._models_used.values())
        self.session.independence_summary.update(
            {
                "models_by_agent": dict(self._models_used),
                "model_family_distinct": len(model_families) > 1,
                "configured_mode": "heterogeneous" if len(model_families) > 1 else "same_model_fallback",
            }
        )

    def _budget_exhausted(self) -> bool:
        calls = int(self.session.budget_summary.get("llm_calls") or 0)
        return calls >= self.max_calls or (time.monotonic() - self.started) >= self.deadline_sec

    def _finish_budget(self, reason: str) -> None:
        self.session.budget_summary["elapsed_ms"] = int((time.monotonic() - self.started) * 1000)
        self.session.budget_summary["termination_reason"] = reason

    @staticmethod
    def _claim_priority(claim: Any) -> float:
        status = {"disputed": 4.0, "needs_search": 3.0, "supported": 2.0, "unsupported": 0.0}.get(claim.status, 1.0)
        return status + len(claim.contradicting_spans) + float(claim.confidence)

    @staticmethod
    def _role_relevance(profile: DebateAgentProfile, claim: Any) -> float:
        text = f"{claim.claim_type} {claim.aspect} {claim.stance} {claim.claim_text}".lower()
        score = float(claim.confidence)
        tokens = set(
            word
            for phrase in [profile.name, profile.analytical_lens, *profile.evidence_obligations]
            for word in str(phrase).lower().replace("_", " ").split()
            if len(word) > 3
        )
        score += sum(0.4 for token in tokens if token in text)
        if any(token in profile.name.lower() for token in ("public", "consumer", "stakeholder")) and claim.stance != "official":
            score += 1.5
        if any(token in profile.name.lower() for token in ("fact", "technical", "data")) and claim.claim_type == "fact":
            score += 1.5
        if any(token in profile.name.lower() for token in ("policy", "ethic", "impact")) and claim.claim_type in {"causal", "risk", "opinion"}:
            score += 1.0
        return score

    @staticmethod
    def _claim_spans(claim_index: Dict[str, Any], claim_ids: Iterable[str]) -> List[str]:
        return list(
            dict.fromkeys(
                span_id
                for claim_id in claim_ids
                if claim_id in claim_index
                for span_id in claim_index[claim_id].supporting_spans + claim_index[claim_id].contradicting_spans
            )
        )

    @staticmethod
    def _merge_verdicts(left: str, right: str) -> str:
        if left == right:
            return left
        values = {left, right}
        if values <= {"accept", "weaken"}:
            return "weaken"
        if "needs_search" in values:
            return "needs_search"
        return "unresolved"

    @staticmethod
    def _final_wording(decision: str, left: JudgeVerdict, right: JudgeVerdict) -> Optional[str]:
        if decision not in {"accept", "weaken"}:
            return None
        if left.decision == "weaken" and left.final_wording:
            return left.final_wording
        if right.decision == "weaken" and right.final_wording:
            return right.final_wording
        return right.final_wording or left.final_wording

    @staticmethod
    def _apply_claim_status(claim: Any, decision: str, final_wording: Optional[str], confidence: float) -> None:
        claim.status = {
            "accept": "supported",
            "weaken": "disputed",
            "reject": "unsupported",
            "needs_search": "needs_search",
            "unresolved": "disputed",
        }.get(decision, claim.status)
        claim.confidence = round(min(claim.confidence, confidence), 4)
        if final_wording and decision in {"accept", "weaken"}:
            claim.claim_text = final_wording

    @staticmethod
    def _strings(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha1("\x1f".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:14]
        return f"{prefix}_{digest}"

    @staticmethod
    def _opening_system_prompt(profile: DebateAgentProfile) -> str:
        return (
            f"You are the independently executed {profile.name} in the Perspective Chamber. "
            "Form positions before seeing peer positions. Use only claim and source-span IDs in the assigned EvidenceView. "
            "Every factual support/challenge/qualification must cite valid span IDs. If evidence is insufficient, abstain or state uncertainty. "
            "Do not infer population opinion from sampled posts. Return only the required JSON object."
        )

    @staticmethod
    def _review_system_prompt(profile: DebateAgentProfile) -> str:
        return (
            f"You are the independent {profile.name} in the Evidence Review Chamber. "
            "Review only the assigned material claims. Challenges must identify a concrete evidence, source-independence, sampling, temporal, or inference problem. "
            "Cite valid span IDs, or use reason code missing_evidence with a typed evidence request. Return only the required JSON object."
        )

    @staticmethod
    def _response_system_prompt(profile: DebateAgentProfile) -> str:
        return (
            f"You are the original {profile.name}; no moderator may answer for you. "
            "Respond to assigned challenges by rebutting with valid spans, revising wording, conceding, abstaining, or requesting evidence. "
            "A revision must preserve sample boundaries and explain what changed. Return only the required JSON object."
        )

    @staticmethod
    def _judge_system_prompt() -> str:
        return (
            "You are an isolated blind claim judge. Agent and model identities are hidden. Apply the claim-mode rubric to validated evidence and argument acts. "
            "Agreement is not correctness. Preserve unresolved empirical disputes and normative tensions; request evidence when decisive support is missing. "
            "Never upgrade repeated discovery into independent confirmation. Return only the required JSON object."
        )
