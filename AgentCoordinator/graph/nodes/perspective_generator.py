"""
PerspectiveGeneratorNode: Selects the 4 deliberation dimensions based on analysis_type.
"""

from __future__ import annotations

from loguru import logger

from ..state import CoordinatorState
from ...utils.perspective_templates import get_perspectives


async def perspective_generator_node(state: CoordinatorState) -> dict:
    """LangGraph node: select perspective templates for this query's analysis_type."""
    analysis_type = state.get("analysis_type", "general")
    perspectives = get_perspectives(analysis_type)
    perspective_names = [name for name, _ in perspectives]

    trace = (
        f"[PerspectiveGen] analysis_type={analysis_type} → "
        f"perspectives: {perspective_names}"
    )
    logger.info(trace)

    return {
        "perspectives": perspective_names,
        "coordinator_trace": [trace],
    }
