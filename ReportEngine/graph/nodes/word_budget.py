"""
Word Budget Node — LangGraph wrapper for WordBudgetNode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def word_budget_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    sections = state["sections"]
    layout_design = state["layout_design"]
    normalized_reports = state["normalized_reports"]
    forum_logs = state.get("forum_logs") or ""
    query = state["query"]
    template_overview = state["template_overview"]

    logger.info("\n[LangGraph:word_budget] Planning chapter word budgets")

    word_plan = agent._run_stage_with_retry(
        "chapter word budget planning",
        lambda: agent.word_budget_node.run(
            sections,
            layout_design,
            normalized_reports,
            forum_logs,
            query,
            template_overview,
        ),
        expected_keys=["chapters", "totalWords", "globalGuidelines"],
        postprocess=agent._normalize_word_plan,
    )

    chapter_targets = {
        entry.get("chapterId"): entry
        for entry in word_plan.get("chapters", [])
        if entry.get("chapterId")
    }

    agent.emit("stage", {
        "stage": "word_plan_ready",
        "chapter_targets": len(chapter_targets),
    })
    agent.emit("progress", {"progress": 20, "message": "Chapter word budget generated"})

    trace = f"[WordBudget] {len(chapter_targets)} chapter targets"
    return {
        "word_plan": word_plan,
        "chapter_targets": chapter_targets,
        "trace_log": [trace],
    }
