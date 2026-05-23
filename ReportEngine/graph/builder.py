"""
Report Agent LangGraph Builder

  START
    → template_selection   (Pick or accept custom template)
    → template_slice       (Parse template into chapter sections)
    → document_layout      (Title / TOC / theme design)
    → word_budget          (Per-chapter word targets)
    → prepare_storage      (Manifest + chapter session directory)
    → [chapter_router]
        ├─ "more"    → process_chapter → [chapter_router]
        └─ "done"    → finalize_report → END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    document_layout_node,
    finalize_report_node,
    prepare_storage_node,
    process_chapter_node,
    template_selection_node,
    template_slice_node,
    word_budget_node,
)
from .state import ReportAgentState

if TYPE_CHECKING:
    from ..agent import ReportAgent


def chapter_router(state: ReportAgentState) -> Literal["more", "done"]:
    """Check if there are remaining chapters to generate."""
    sections = state.get("sections") or []
    idx = state.get("chapter_index", 0)
    if not sections:
        return "done"
    if idx >= len(sections):
        return "done"
    return "more"


def build_report_agent_graph(agent: "ReportAgent"):
    """
    Build and compile Report Agent LangGraph.

    Returns:
        CompiledGraph — can call .invoke(state) / .ainvoke(state)
    """
    graph = StateGraph(ReportAgentState)

    def _template_selection(s: ReportAgentState) -> dict:
        return template_selection_node(agent, s)

    def _template_slice(s: ReportAgentState) -> dict:
        return template_slice_node(agent, s)

    def _document_layout(s: ReportAgentState) -> dict:
        return document_layout_node(agent, s)

    def _word_budget(s: ReportAgentState) -> dict:
        return word_budget_node(agent, s)

    def _prepare_storage(s: ReportAgentState) -> dict:
        return prepare_storage_node(agent, s)

    def _process_chapter(s: ReportAgentState) -> dict:
        return process_chapter_node(agent, s)

    def _finalize_report(s: ReportAgentState) -> dict:
        return finalize_report_node(agent, s)

    graph.add_node("template_selection", _template_selection)
    graph.add_node("template_slice", _template_slice)
    graph.add_node("document_layout", _document_layout)
    graph.add_node("word_budget", _word_budget)
    graph.add_node("prepare_storage", _prepare_storage)
    graph.add_node("process_chapter", _process_chapter)
    graph.add_node("finalize_report", _finalize_report)

    graph.add_edge(START, "template_selection")
    graph.add_edge("template_selection", "template_slice")
    graph.add_edge("template_slice", "document_layout")
    graph.add_edge("document_layout", "word_budget")
    graph.add_edge("word_budget", "prepare_storage")

    graph.add_conditional_edges(
        "prepare_storage",
        chapter_router,
        {"more": "process_chapter", "done": "finalize_report"},
    )

    graph.add_conditional_edges(
        "process_chapter",
        chapter_router,
        {"more": "process_chapter", "done": "finalize_report"},
    )

    graph.add_edge("finalize_report", END)

    return graph.compile()
