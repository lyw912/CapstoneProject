"""
LangGraph builder for AgentCoordinator.

Graph topology (from design document Section 10.1):
  START → [query_agent || media_agent] → data_bridge → divergence_matrix
        → perspective_gen → deliberation → [gap_detector]
          → (need_search) → targeted_search → deliberation (back-edge)
          → (sufficient/max_rounds) → echo_chamber → fact_opinion
        → platform_interpret → synthesis → report_agent → END
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .nodes.query_agent_node import query_agent_node
from .nodes.media_agent_node import media_agent_node
from .nodes.data_bridge_node import data_bridge_node
from .nodes.divergence_matrix_node import divergence_matrix_node
from .nodes.perspective_generator import perspective_generator_node
from .nodes.deliberation_engine import deliberation_engine_node
from .nodes.gap_detector import gap_detector_router
from .nodes.targeted_search_node import targeted_search_node
from .nodes.echo_chamber_detector import echo_chamber_detector_node
from .nodes.fact_opinion_separator import fact_opinion_separator_node
from .nodes.platform_interpreter import platform_interpreter_node
from .nodes.synthesis_node import synthesis_node
from .nodes.report_agent_node import report_agent_node
from .state import CoordinatorState


def build_coordinator_graph(use_checkpointing: bool = True, checkpointer=None) -> StateGraph:
    """Build and compile the AgentCoordinator LangGraph state machine.

    Args:
        use_checkpointing: If True (default), attach a MemorySaver checkpointer so
            the pipeline can resume on failure via thread_id-based checkpoints.
        checkpointer: Optional custom checkpointer instance. If None and
            use_checkpointing=True, a fresh MemorySaver() is created.
    """
    graph = StateGraph(CoordinatorState)

    # ── Phase 0: Parallel agent execution ──
    graph.add_node("query_agent", query_agent_node)
    graph.add_node("media_agent", media_agent_node)

    # ── Phase 1: Data bridging + divergence matrix ──
    graph.add_node("data_bridge", data_bridge_node)
    graph.add_node("divergence_compute", divergence_matrix_node)

    # ── Phase 2: Deliberation ──
    graph.add_node("perspective_gen", perspective_generator_node)
    graph.add_node("deliberation", deliberation_engine_node)
    graph.add_node("targeted_search", targeted_search_node)

    # ── Phase 3: Bias correction + fact separation ──
    graph.add_node("echo_chamber", echo_chamber_detector_node)
    graph.add_node("fact_opinion", fact_opinion_separator_node)

    # ── Phase 4: Platform interpretation + synthesis + report ──
    graph.add_node("platform_interpret", platform_interpreter_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_agent", report_agent_node)

    # ── Edges ──

    # Fan-out: START → both agents in parallel
    graph.add_edge(START, "query_agent")
    graph.add_edge(START, "media_agent")

    # Fan-in: both agents → data_bridge (LangGraph waits for both via superstep)
    graph.add_edge("query_agent", "data_bridge")
    graph.add_edge("media_agent", "data_bridge")

    # Sequential Phase 1
    graph.add_edge("data_bridge", "divergence_compute")
    graph.add_edge("divergence_compute", "perspective_gen")
    graph.add_edge("perspective_gen", "deliberation")

    # Conditional CRAG back-edge
    graph.add_conditional_edges(
        "deliberation",
        gap_detector_router,
        {
            "sufficient": "echo_chamber",
            "need_search": "targeted_search",
            "max_rounds": "echo_chamber",
        },
    )
    graph.add_edge("targeted_search", "deliberation")  # back-edge for CRAG loop

    # Phase 3 sequential
    graph.add_edge("echo_chamber", "fact_opinion")

    # Phase 4 sequential
    graph.add_edge("fact_opinion", "platform_interpret")
    graph.add_edge("platform_interpret", "synthesis")
    graph.add_edge("synthesis", "report_agent")
    graph.add_edge("report_agent", END)

    # Determine checkpointer to use
    if use_checkpointing:
        if checkpointer is None:
            checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()
