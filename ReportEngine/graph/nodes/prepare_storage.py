"""
Prepare Storage Node — manifest, generation context, chapter session directory
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def prepare_storage_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    query = state["query"]
    report_id = state["report_id"]
    template_result = state["template_result"]
    layout_design = state["layout_design"]
    word_plan = state["word_plan"]
    chapter_targets = state["chapter_targets"]
    template_overview = state["template_overview"]
    normalized_reports = state["normalized_reports"]
    forum_logs = state.get("forum_logs") or ""

    logger.info("\n[LangGraph:prepare_storage] Initializing chapter storage session")

    generation_context = agent._build_generation_context(
        query,
        normalized_reports,
        forum_logs,
        template_result,
        layout_design,
        chapter_targets,
        word_plan,
        template_overview,
    )

    manifest_meta = {
        "query": query,
        "title": layout_design.get("title")
        or (
            f"{query} - Sentiment Insight Report"
            if query
            else template_result.get("template_name")
        ),
        "subtitle": layout_design.get("subtitle"),
        "tagline": layout_design.get("tagline"),
        "templateName": template_result.get("template_name"),
        "selectionReason": template_result.get("selection_reason"),
        "themeTokens": generation_context.get("theme_tokens", {}),
        "toc": {
            "depth": 3,
            "autoNumbering": True,
            "title": layout_design.get("tocTitle") or "Table of Contents",
        },
        "hero": layout_design.get("hero"),
        "layoutNotes": layout_design.get("layoutNotes"),
        "wordPlan": {
            "totalWords": word_plan.get("totalWords"),
            "globalGuidelines": word_plan.get("globalGuidelines"),
        },
        "templateOverview": template_overview,
    }
    if layout_design.get("themeTokens"):
        manifest_meta["themeTokens"] = layout_design["themeTokens"]
    if layout_design.get("tocPlan"):
        manifest_meta["toc"]["customEntries"] = layout_design["tocPlan"]

    run_dir = agent.chapter_storage.start_session(report_id, manifest_meta)
    agent._persist_planning_artifacts(run_dir, layout_design, word_plan, template_overview)

    agent.emit("stage", {"stage": "storage_ready", "run_dir": str(run_dir)})

    trace = f"[PrepareStorage] session at {run_dir}"
    return {
        "generation_context": generation_context,
        "manifest_meta": manifest_meta,
        "run_dir": str(run_dir),
        "trace_log": [trace],
    }
