"""
Media Agent LangGraph State Definition

Consistent with QueryEngine, uses TypedDict to pass state between nodes;
pipeline_state is the existing dataclass State, carrying paragraphs and search traces.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from ..state.state import State


class MediaAgentState(TypedDict, total=False):
    """Media Deep Research LangGraph runtime state."""

    original_query: str
    paragraph_index: int
    max_reflections: int

    pipeline_state: State

    final_report: Optional[str]

    trace_log: Annotated[List[str], operator.add]
    error_log: Annotated[List[str], operator.add]
