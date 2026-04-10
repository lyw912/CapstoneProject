"""
最终报告节点 — 对应原 _generate_final_report
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import MediaAgentState

if TYPE_CHECKING:
    from ...agent import DeepSearchAgent


def finalize_report_node(agent: DeepSearchAgent, state: MediaAgentState) -> dict:
    logger.info(f"\n[LangGraph:finalize_report] 生成最终报告...")

    ps = state["pipeline_state"]

    report_data = []
    for paragraph in ps.paragraphs:
        report_data.append({
            "title": paragraph.title,
            "paragraph_latest_state": paragraph.research.latest_summary,
        })

    try:
        final_report = agent.report_formatting_node.run(report_data)
    except Exception as e:
        logger.error(f"LLM格式化失败，使用备用方法: {str(e)}")
        final_report = agent.report_formatting_node.format_report_manually(
            report_data, ps.report_title
        )

    ps.final_report = final_report
    ps.mark_completed()

    logger.info("最终报告生成完成")
    return {
        "pipeline_state": ps,
        "final_report": final_report,
        "trace_log": ["[FinalizeReport] 完成"],
    }
