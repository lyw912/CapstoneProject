"""
Media Agent LangGraph 状态定义

与 QueryEngine 一致，使用 TypedDict 在节点间传递状态；
pipeline_state 为现有 dataclass State，承载段落与搜索轨迹。
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from ..state.state import State


class MediaAgentState(TypedDict, total=False):
    """Media 深度研究 LangGraph 运行时状态。"""

    original_query: str
    paragraph_index: int
    max_reflections: int

    pipeline_state: State

    final_report: Optional[str]

    trace_log: Annotated[List[str], operator.add]
    error_log: Annotated[List[str], operator.add]
