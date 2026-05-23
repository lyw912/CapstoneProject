"""
Finalize Report Node — stitch IR, render HTML, persist outputs
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from ..state import ReportAgentState

if TYPE_CHECKING:
    from ...agent import ReportAgent


def finalize_report_node(agent: "ReportAgent", state: ReportAgentState) -> dict:
    report_id = state["report_id"]
    manifest_meta = state["manifest_meta"]
    chapters = state.get("chapters") or []

    logger.info(f"\n[LangGraph:finalize_report] Compiling {len(chapters)} chapters")

    document_ir = agent.document_composer.build_document(
        report_id,
        manifest_meta,
        chapters,
    )
    agent.emit("stage", {"stage": "chapters_compiled", "chapter_count": len(chapters)})

    html_report = agent.renderer.render(document_ir)
    agent.emit("stage", {"stage": "html_rendered", "html_length": len(html_report)})

    agent.state.html_content = html_report
    agent.state.mark_completed()

    saved_files: dict = {}
    if state.get("save_report", True):
        saved_files = agent._save_report(html_report, document_ir, report_id)
        agent.emit("stage", {"stage": "report_saved", "files": saved_files})

    generation_time = 0.0
    if agent._report_start_time:
        generation_time = (datetime.now() - agent._report_start_time).total_seconds()
    agent.state.metadata.generation_time = generation_time

    logger.info(f"Report generation complete, elapsed: {generation_time:.2f} seconds")
    agent.emit("metrics", {"generation_seconds": generation_time})

    trace = f"[FinalizeReport] html={len(html_report)} chars"
    return {
        "html_content": html_report,
        "document_ir": document_ir,
        "saved_files": saved_files,
        "generation_time": generation_time,
        "trace_log": [trace],
    }
