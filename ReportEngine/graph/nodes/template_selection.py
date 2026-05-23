"""
Template Selection Node — LangGraph wrapper for TemplateSelectionNode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def template_selection_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    query = state["query"]
    reports = state.get("reports") or []
    forum_logs = state.get("forum_logs") or ""
    custom_template = state.get("custom_template") or ""

    logger.info(f"\n[LangGraph:template_selection] Selecting template for: {query!r}")

    template_result = agent._select_template(query, reports, forum_logs, custom_template)
    template_result = agent._ensure_mapping(
        template_result,
        "template selection result",
        expected_keys=["template_name", "template_content"],
    )
    agent.state.metadata.template_used = template_result.get("template_name", "")

    agent.emit("stage", {
        "stage": "template_selected",
        "template": template_result.get("template_name"),
        "reason": template_result.get("selection_reason"),
    })
    agent.emit("progress", {"progress": 10, "message": "Template selection complete"})

    trace = f"[TemplateSelection] {template_result.get('template_name')}"
    return {
        "template_result": template_result,
        "trace_log": [trace],
    }
