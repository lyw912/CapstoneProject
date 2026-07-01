"""
Finalize Report Node — corresponds to original _generate_final_report
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import MediaAgentState

if TYPE_CHECKING:
    from ...agent import DeepSearchAgent


def finalize_report_node(agent: DeepSearchAgent, state: MediaAgentState) -> dict:
    logger.info(f"\n[LangGraph:finalize_report] Generating final report...")

    ps = state["pipeline_state"]

    report_data = []
    for paragraph in ps.paragraphs:
        summary = paragraph.research.latest_summary
        if not summary:
            continue
        report_data.append({
            "title": paragraph.title,
            "paragraph_latest_state": summary,
        })

    if not report_data:
        raise RuntimeError("[FinalizeReport] No paragraph summaries available for final report")

    completed = len(report_data)
    total = len(ps.paragraphs)
    if completed < total:
        logger.warning(
            f"[FinalizeReport] Building partial report from {completed}/{total} paragraphs"
        )

    use_llm_format = bool(getattr(agent.config, "MEDIA_USE_LLM_REPORT_FORMAT", False))

    if use_llm_format:
        try:
            final_report = agent.report_formatting_node.run(report_data)
        except Exception as e:
            logger.error(f"LLM formatting failed, using fallback method: {str(e)}")
            final_report = agent.report_formatting_node.format_report_manually(
                report_data, ps.report_title
            )
    else:
        logger.info("Assembling final report without LLM (MEDIA_USE_LLM_REPORT_FORMAT=false)")
        final_report = agent.report_formatting_node.format_report_manually(
            report_data, ps.report_title
        )

    ps.final_report = final_report
    ps.mark_completed()

    logger.info("Final report generation completed")
    return {
        "pipeline_state": ps,
        "final_report": final_report,
        "trace_log": ["[FinalizeReport] Completed"],
    }
