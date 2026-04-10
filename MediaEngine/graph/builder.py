"""
Media Agent LangGraph 构建

  START
    → report_structure   （生成报告段落结构）
    → [paragraph_router]
        ├─ "more"    → process_paragraph → [paragraph_router]
        └─ "done"    → finalize_report → END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, START, StateGraph

from .nodes import finalize_report_node, process_paragraph_node, report_structure_node
from .state import MediaAgentState

if TYPE_CHECKING:
    from ..agent import DeepSearchAgent


def paragraph_router(state: MediaAgentState) -> Literal["more", "done"]:
    """是否还有待处理段落。"""
    ps = state.get("pipeline_state")
    idx = state.get("paragraph_index", 0)
    if not ps or not ps.paragraphs:
        return "done"
    if idx >= len(ps.paragraphs):
        return "done"
    return "more"


def build_media_agent_graph(agent: DeepSearchAgent):
    """
    构建并编译 Media Agent LangGraph。

    Returns:
        CompiledGraph — 可调用 .invoke(state) / .ainvoke(state)
    """
    graph = StateGraph(MediaAgentState)

    def _report_structure(s: MediaAgentState) -> dict:
        return report_structure_node(agent, s)

    def _process_paragraph(s: MediaAgentState) -> dict:
        return process_paragraph_node(agent, s)

    def _finalize_report(s: MediaAgentState) -> dict:
        return finalize_report_node(agent, s)

    graph.add_node("report_structure", _report_structure)
    graph.add_node("process_paragraph", _process_paragraph)
    graph.add_node("finalize_report", _finalize_report)

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
