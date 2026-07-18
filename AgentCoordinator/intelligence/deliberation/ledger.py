"""Append-only argument ledger with evidence-reference validation."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Set

from ..contracts import (
    AgentPosition,
    ArgumentAct,
    DebateSession,
    EvidenceGraph,
    JudgeVerdict,
    PositionRevision,
    ProtocolFailure,
)


class InvalidArgumentReference(ValueError):
    pass


class ArgumentLedger:
    def __init__(self, session: DebateSession):
        self.session = session

    def add_position(self, position: AgentPosition, graph: EvidenceGraph, allowed_claim_ids: Iterable[str]) -> None:
        self._validate_claim(position.claim_id, graph, set(allowed_claim_ids))
        self._validate_spans(position.evidence_span_ids, graph, allow_empty=position.stance == "abstain")
        if any(item.position_id == position.position_id for item in self.session.positions):
            raise InvalidArgumentReference(f"Duplicate position_id: {position.position_id}")
        self.session.positions.append(position)

    def add_act(self, act: ArgumentAct, graph: EvidenceGraph, allowed_claim_ids: Iterable[str]) -> None:
        self._validate_claim(act.target_claim_id, graph, set(allowed_claim_ids))
        allow_empty = act.act_type in {"request_evidence", "abstain", "concede"} or "missing_evidence" in act.reason_codes
        self._validate_spans(act.evidence_span_ids, graph, allow_empty=allow_empty)
        if any(item.act_id == act.act_id for item in self.session.argument_acts):
            raise InvalidArgumentReference(f"Duplicate act_id: {act.act_id}")
        self.session.argument_acts.append(act)

    def add_revision(self, revision: PositionRevision, graph: EvidenceGraph) -> None:
        if revision.previous_position_id not in {item.position_id for item in self.session.positions}:
            raise InvalidArgumentReference(f"Unknown previous_position_id: {revision.previous_position_id}")
        if not set(revision.triggering_act_ids).issubset({item.act_id for item in self.session.argument_acts}):
            raise InvalidArgumentReference("Revision references an unknown triggering act")
        self._validate_spans(revision.evidence_span_ids, graph, allow_empty=revision.revision_type in {"concede", "abstain"})
        self.session.revisions.append(revision)

    def add_verdict(self, verdict: JudgeVerdict, graph: EvidenceGraph) -> None:
        self._validate_claim(verdict.claim_id, graph, {verdict.claim_id})
        self._validate_spans(verdict.evidence_span_ids, graph, allow_empty=verdict.decision in {"reject", "needs_search", "unresolved"})
        if not set(verdict.decisive_act_ids).issubset({item.act_id for item in self.session.argument_acts}):
            raise InvalidArgumentReference("Verdict references an unknown argument act")
        self.session.verdicts.append(verdict)

    def failure(
        self,
        phase: str,
        agent_id: str,
        failure_type: str,
        message: str,
        claim_id: Optional[str] = None,
        retryable: bool = False,
    ) -> ProtocolFailure:
        digest = hashlib.sha1(
            f"{phase}\x1f{agent_id}\x1f{failure_type}\x1f{message}\x1f{len(self.session.protocol_failures)}".encode("utf-8")
        ).hexdigest()[:12]
        failure = ProtocolFailure(
            failure_id=f"pf_{digest}",
            phase=phase,
            agent_id=agent_id,
            failure_type=failure_type,
            message=str(message)[:1000],
            claim_id=claim_id,
            retryable=retryable,
        )
        self.session.protocol_failures.append(failure)
        return failure

    @staticmethod
    def _validate_claim(claim_id: str, graph: EvidenceGraph, allowed: Set[str]) -> None:
        if claim_id not in graph.claim_index():
            raise InvalidArgumentReference(f"Unknown claim_id: {claim_id}")
        if allowed and claim_id not in allowed:
            raise InvalidArgumentReference(f"Claim is outside the assigned evidence view: {claim_id}")

    @staticmethod
    def _validate_spans(span_ids: List[str], graph: EvidenceGraph, allow_empty: bool) -> None:
        if not span_ids and not allow_empty:
            raise InvalidArgumentReference("Evidence-bound argument requires at least one source span")
        unknown = set(span_ids) - set(graph.span_index())
        if unknown:
            raise InvalidArgumentReference(f"Unknown evidence span IDs: {sorted(unknown)}")
