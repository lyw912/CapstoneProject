"""
Report Agent LangGraph State Definitions

TypedDict structures for passing state between LangGraph nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from ..core import TemplateSection


class ReportAgentState(TypedDict, total=False):
    """Complete runtime state for Report Agent LangGraph."""

    # === Input ===
    query: str
    reports: List[Any]
    forum_logs: str
    custom_template: str
    save_report: bool
    report_id: str

    # === Normalized inputs ===
    normalized_reports: Dict[str, str]

    # === Template pipeline ===
    template_result: Dict[str, Any]
    sections: List[TemplateSection]
    template_overview: Dict[str, Any]

    # === Planning ===
    layout_design: Dict[str, Any]
    word_plan: Dict[str, Any]
    chapter_targets: Dict[str, Dict[str, Any]]
    generation_context: Dict[str, Any]
    manifest_meta: Dict[str, Any]
    run_dir: str

    # === Chapter loop ===
    chapter_index: int
    chapters: Annotated[List[Dict[str, Any]], operator.add]

    # === Output ===
    html_content: str
    document_ir: Dict[str, Any]
    saved_files: Dict[str, Any]
    generation_time: float

    # === Monitoring ===
    trace_log: Annotated[List[str], operator.add]
    error_message: str
