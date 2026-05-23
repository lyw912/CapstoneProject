"""
Template Slice Node — parse template Markdown into chapter sections
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def template_slice_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    template_result = state["template_result"]
    template_text = template_result.get("template_content", "")

    logger.info("\n[LangGraph:template_slice] Slicing template into chapters")

    sections = agent._slice_template(template_text)
    if not sections:
        raise ValueError("Template cannot be parsed into chapters, please check template content.")

    template_overview = agent._build_template_overview(template_text, sections)

    agent.emit("stage", {"stage": "template_sliced", "section_count": len(sections)})

    trace = f"[TemplateSlice] {len(sections)} sections"
    return {
        "sections": sections,
        "template_overview": template_overview,
        "chapter_index": 0,
        "chapters": [],
        "trace_log": [trace],
    }
