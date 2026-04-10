"""
报告结构节点 — 对应原 DeepSearchAgent._generate_report_structure
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..state import MediaAgentState
from ...nodes import ReportStructureNode
from ...state.state import State

if TYPE_CHECKING:
    from ...agent import DeepSearchAgent


def report_structure_node(agent: DeepSearchAgent, state: MediaAgentState) -> dict:
    query = state["original_query"]
    logger.info(f"\n[LangGraph:report_structure] 生成报告结构: {query!r}")

    node = ReportStructureNode(agent.llm_client, query)
    st = State()
    new_state = node.mutate_state(state=st)

    trace = (
        f"[ReportStructure] 共 {len(new_state.paragraphs)} 个段落: "
        + ", ".join(p.title for p in new_state.paragraphs[:8])
        + ("..." if len(new_state.paragraphs) > 8 else "")
    )
    logger.info(trace)

    return {
        "pipeline_state": new_state,
        "paragraph_index": 0,
        "trace_log": [trace],
    }
