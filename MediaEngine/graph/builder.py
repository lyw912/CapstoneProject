"""
Media Agent LangGraph Builder

  START
    → report_structure   (Generate report paragraph structure)
    → process_paragraph(s)  (sequential loop OR parallel batch)
    → finalize_report → END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import finalize_report_node, process_all_paragraphs_node, process_paragraph_node, report_structure_node
from .state import MediaAgentState

if TYPE_CHECKING:
    from ..agent import DeepSearchAgent


def paragraph_router(state: MediaAgentState) -> Literal["more", "done"]:
    """Check if there are remaining paragraphs to process (sequential mode only)."""
    ps = state.get("pipeline_state")
    idx = state.get("paragraph_index", 0)
    if not ps or not ps.paragraphs:
        return "done"
    if idx >= len(ps.paragraphs):
        return "done"
    return "more"


def _use_parallel_paragraphs(agent: "DeepSearchAgent") -> bool:
    workers = int(getattr(agent.config, "MEDIA_PARAGRAPH_WORKERS", 1) or 1)
    return workers > 1


def build_media_agent_graph(agent: DeepSearchAgent):
    """
    Build and compile Media Agent LangGraph.

    Returns:
        CompiledGraph — can call .invoke(state) / .ainvoke(state)
    """
    graph = StateGraph(MediaAgentState)

    def _report_structure(s: MediaAgentState) -> dict:
        return report_structure_node(agent, s)

    def _process_paragraph(s: MediaAgentState) -> dict:
        return process_paragraph_node(agent, s)

    def _process_all_paragraphs(s: MediaAgentState) -> dict:
        return process_all_paragraphs_node(agent, s)

    def _finalize_report(s: MediaAgentState) -> dict:
        return finalize_report_node(agent, s)

    graph.add_node("report_structure", _report_structure)
    graph.add_node("finalize_report", _finalize_report)

    if _use_parallel_paragraphs(agent):
        graph.add_node("process_all_paragraphs", _process_all_paragraphs)
        graph.add_edge(START, "report_structure")
        graph.add_edge("report_structure", "process_all_paragraphs")
        graph.add_edge("process_all_paragraphs", "finalize_report")
    else:
        graph.add_node("process_paragraph", _process_paragraph)
        graph.add_edge(START, "report_structure")
        graph.add_conditional_edges(
            "report_structure",
            paragraph_router,
            {"more": "process_paragraph", "done": "finalize_report"},
        )
        graph.add_conditional_edges(
            "process_paragraph",
            paragraph_router,
            {"more": "process_paragraph", "done": "finalize_report"},
        )

    graph.add_edge("finalize_report", END)

    return graph.compile()
