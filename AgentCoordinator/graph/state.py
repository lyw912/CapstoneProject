"""
CoordinatorState: LangGraph state definition for AgentCoordinator.

All phases' state fields are defined here.
Uses Annotated[List, operator.add] for reducer fields that accumulate across nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


class AgentRunResult(TypedDict):
    """Wrapper for individual agent outputs."""
    agent_name: str
    success: bool
    output: Optional[Dict]        # QueryAgentOutput for query_agent
    text_output: Optional[str]    # Markdown text for media_agent
    error: Optional[str]
    duration_seconds: float


class BridgedProposition(TypedDict):
    """Unified proposition format bridged from heterogeneous agent outputs."""
    prop_id: str
    content: str
    source_agent: str             # "query_agent" | "media_agent"
    stance: Optional[str]         # stance_label if available
    confidence: float
    evidence_urls: List[str]
    platform: Optional[str]


class DeliberationRound(TypedDict):
    """Record of one phase of the deliberation process."""
    phase: str                    # "independent" | "cross_examination" | "synthesis"
    perspectives: List[Dict]      # Per-perspective analysis results
    consensus_points: List[str]
    dissent_points: List[str]
    raw_llm_output: Optional[str]


class SearchGap(TypedDict):
    """Detected information gap requiring supplementary search."""
    gap_id: str
    description: str
    target_query: str
    target_source: str            # "mindspider_db" | "tavily" | "broad_topic"
    rationale: str
    priority: int                 # 1=high, 2=medium, 3=low


class CoordinatorState(TypedDict):
    """Complete LangGraph state for AgentCoordinator pipeline."""

    # === Input ===
    query: str
    analysis_type: str            # event/brand/policy/technology/general

    # === Phase 0: Agent execution results ===
    query_run: Optional[AgentRunResult]
    media_run: Optional[AgentRunResult]
    agent_errors: Annotated[List[str], operator.add]

    # === Phase 1: Data bridging + divergence matrix ===
    bridged_propositions: Optional[List[BridgedProposition]]
    divergence_matrix: Optional[Dict]          # {(source_a, source_b): delta_value}
    divergence_hotspots: Optional[List[str]]   # Divergence hotspot descriptions

    # === Phase 2: Deliberation ===
    perspectives: Optional[List[str]]          # Selected perspective names
    deliberation_rounds: Optional[List[DeliberationRound]]
    deliberation_consensus: Optional[List[str]]
    deliberation_dissents: Optional[List[str]]

    # === Phase 2.5: Gap filling (CRAG-driven) ===
    search_gaps: Optional[List[SearchGap]]
    supplementary_results: Optional[List[Dict]]
    search_rounds: int                         # Max 1 supplementary search round

    # === Phase 3: Echo chamber detection + fact-opinion separation ===
    echo_warnings: Optional[List[str]]
    silent_majority_hypothesis: Optional[str]
    verified_facts: Optional[List[Dict]]
    opinions_sentiments: Optional[List[Dict]]
    analytical_frameworks: Optional[List[Dict]]

    # === Phase 4: Synthesis + report ===
    platform_interpretations: Optional[Dict]  # {platform: interpretation_text}
    synthesis_context: Optional[Dict]         # Full context passed to ReportAgent
    synthesis_confidence: float
    report_output: Optional[str]              # Final HTML/Markdown report

    # === Full-pipeline tracing ===
    coordinator_trace: Annotated[List[str], operator.add]
