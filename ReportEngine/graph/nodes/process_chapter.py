"""
Process Chapter Node — generate one chapter with retry / streaming (loop body)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def process_chapter_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    sections = state["sections"]
    chapter_index = state.get("chapter_index", 0)
    section = sections[chapter_index]
    total_chapters = len(sections)

    logger.info(
        f"\n[LangGraph:process_chapter] Chapter {chapter_index + 1}/{total_chapters}: {section.title}"
    )

    run_dir = Path(state["run_dir"])
    generation_context = state["generation_context"]

    chapter_payload, attempt, fallback_used = agent._generate_single_chapter(
        section,
        generation_context,
        run_dir,
    )

    completed = chapter_index + 1
    chapter_progress = 20 + round(80 * completed / total_chapters)
    agent.emit("progress", {
        "progress": chapter_progress,
        "message": f"Chapter {completed}/{total_chapters} completed",
    })

    completion_status = {
        "chapterId": section.chapter_id,
        "title": section.title,
        "status": "completed",
        "attempt": attempt,
    }
    if fallback_used:
        completion_status["warning"] = "content_sparse_fallback"
        completion_status["warningMessage"] = agent._CONTENT_SPARSE_WARNING_TEXT
    agent.emit("chapter_status", completion_status)

    trace = f"[ProcessChapter] {section.chapter_id} completed (attempt {attempt})"
    return {
        "chapters": [chapter_payload],
        "chapter_index": chapter_index + 1,
        "trace_log": [trace],
    }
