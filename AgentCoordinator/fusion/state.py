"""Parent LangGraph state for specialist fusion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class FusionState(TypedDict, total=False):
    query: str
    run_id: str
    mode: str
    target_entity: str
    analysis_type: str
    investigation_brief: Dict[str, Any]
    started_at: float
    blackboard_version: int
    core_version: int
    evidence_graph_summary: Dict[str, Any]
    pending_tasks: List[Any]
    pending_contribution_count: int
    debate_follow_up_claim_ids: List[str]
    debate_status: str
    debate_summary: Dict[str, Any]
    research_round: int
    max_research_rounds: int
    provider_diagnostics: List[Any]
    research_trace: List[Any]
    progress_state: Dict[str, Any]
    artifact_ref: Optional[str]
    artifact_ready: bool
