"""
Report Engine.

An intelligent report generation AI agent implementation that aggregates
Markdown and forum discussions from Query/Media sub-engines,
ultimately producing structured HTML reports (LangGraph orchestration).
"""

from .agent import ReportAgent, create_agent
from .graph import ReportAgentState, build_report_agent_graph

__version__ = "1.0.0"
__author__ = "Report Engine Team"

__all__ = [
    "ReportAgent",
    "create_agent",
    "ReportAgentState",
    "build_report_agent_graph",
]
