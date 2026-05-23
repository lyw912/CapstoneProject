"""
ReportAgentNode: Generates academic-style evidence-traced markdown report.

Uses AcademicReportGenerator to transform the full coordinator state into a
structured research-paper-style report with:
  - Abstract, Methodology, Findings, Conclusions, Appendices
  - Cited social media posts with clickable links
  - Metric definitions (CSSD, SCS, TrustScore, Shannon Entropy)
  - Fact-opinion separation, platform interpretations
  - Placeholder markers for Phase 3 visualizations
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from ..state import CoordinatorState

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_generator_input(state: CoordinatorState) -> dict:
    """Assemble AcademicReportGenerator input via the canonical schema builder."""
    from ...coordinator_output_schema import build_coordinator_output

    result = {
        "synthesis_context": state.get("synthesis_context") or {},
        "divergence_matrix": state.get("divergence_matrix") or {},
        "divergence_hotspots": state.get("divergence_hotspots") or [],
        "deliberation_consensus": state.get("deliberation_consensus") or [],
        "deliberation_dissents": state.get("deliberation_dissents") or [],
        "echo_warnings": state.get("echo_warnings") or [],
        "verified_facts": state.get("verified_facts") or [],
        "platform_interpretations": state.get("platform_interpretations") or {},
        "coordinator_trace": state.get("coordinator_trace") or [],
        "agent_errors": state.get("agent_errors") or [],
        "synthesis_confidence": state.get("synthesis_confidence", 0.5),
    }
    return build_coordinator_output(
        result=result,
        query=state["query"],
        duration_seconds=0.0,
    )


async def report_agent_node(state: CoordinatorState) -> dict:
    """LangGraph node: generate academic-style evidence-traced markdown report."""
    t0 = time.time()

    from ...academic_report_generator import generate_academic_report

    generator_input = _build_generator_input(state)
    report = generate_academic_report(generator_input)

    duration = time.time() - t0
    trace = (
        f"[ReportAgentNode] academic_report — {len(report)} chars in {duration:.1f}s"
    )
    logger.info(trace)

    return {
        "report_output": report,
        "coordinator_trace": [trace],
    }


