"""
Document Layout Node — LangGraph wrapper for DocumentLayoutNode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def document_layout_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    sections = state["sections"]
    template_result = state["template_result"]
    template_text = template_result.get("template_content", "")
    normalized_reports = state["normalized_reports"]
    forum_logs = state.get("forum_logs") or ""
    query = state["query"]
    template_overview = state["template_overview"]

    logger.info("\n[LangGraph:document_layout] Designing document layout")

    layout_design = agent._run_stage_with_retry(
        "document design",
        lambda: agent.document_layout_node.run(
            sections,
            template_text,
            normalized_reports,
            forum_logs,
            query,
            template_overview,
        ),
        expected_keys=["title", "hero", "tocPlan", "tocTitle"],
    )

    agent.emit("stage", {
        "stage": "layout_designed",
        "title": layout_design.get("title"),
        "toc": layout_design.get("tocTitle"),
    })
    agent.emit("progress", {"progress": 15, "message": "Document title/TOC design complete"})

    trace = f"[DocumentLayout] {layout_design.get('title', '')}"
    return {
        "layout_design": layout_design,
        "trace_log": [trace],
    }
