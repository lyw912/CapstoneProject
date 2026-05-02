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
    """Assemble the dict expected by AcademicReportGenerator from state fields."""
    query_run = state.get("query_run")
    media_run = state.get("media_run")
    qa_output = query_run.get("output") if query_run else None
    media_text = media_run.get("text_output") if media_run else None

    top_sources = []
    social_sentiment = None
    stance_distribution = {}
    coverage_score = 0.0
    total_sources = 0

    if qa_output:
        top_sources = sorted(
            qa_output.get("sources", []),
            key=lambda x: x.get("trust_score", 0),
            reverse=True,
        )[:15]
        social_sentiment = qa_output.get("social_sentiment")
        stance_distribution = qa_output.get("stance_distribution", {})
        coverage_score = qa_output.get("coverage_score", 0.0)
        total_sources = qa_output.get("total_sources_kept", 0)

    # Build deliberation object from state
    deliberation = {
        "analysis_type": state.get("analysis_type", "general"),
        "perspectives_used": state.get("perspectives") or [],
        "phases": [],
        "final_consensus": state.get("deliberation_consensus") or [],
        "final_dissents": state.get("deliberation_dissents") or [],
        "confidence": state.get("synthesis_confidence", 0.5),
    }
    for r in (state.get("deliberation_rounds") or []):
        deliberation["phases"].append({
            "phase": r.get("phase", ""),
            "summary": r.get("raw_llm_output", "") or "",
            "consensus_points": r.get("consensus_points", []),
            "dissent_points": r.get("dissent_points", []),
        })

    # Build divergence matrix
    div_matrix = state.get("divergence_matrix") or {}
    div_hotspots = state.get("divergence_hotspots") or []
    max_pair, max_val, min_pair, min_val = "", 0.0, "", 1.0
    if div_matrix:
        max_pair = max(div_matrix, key=div_matrix.get)
        max_val = div_matrix[max_pair]
        min_pair = min(div_matrix, key=div_matrix.get)
        min_val = div_matrix[min_pair]

    synthesis_ctx = state.get("synthesis_context") or {}

    return {
        "schema_version": "1.0",
        "query": state["query"],
        "analysis_type": state.get("analysis_type", "general"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_duration_seconds": 0,  # filled by coordinator
        "divergence_matrix": {
            "pairs": div_matrix,
            "hotspots": div_hotspots,
            "max_divergence": {"pair": max_pair, "value": max_val},
            "min_divergence": {"pair": min_pair, "value": min_val},
        },
        "deliberation": deliberation,
        "gap_filling": {
            "rounds_performed": state.get("search_rounds", 0),
            "gaps_detected": [
                {"description": g.get("description", ""), "source": g.get("target_source", "")}
                for g in (state.get("search_gaps") or [])
            ],
            "results_found": len(state.get("supplementary_results") or []),
        },
        "platform_interpretations": state.get("platform_interpretations") or {},
        "bias_analysis": {
            "echo_warnings": state.get("echo_warnings") or [],
            "silent_majority_hypothesis": state.get("silent_majority_hypothesis"),
        },
        "fact_opinion_separation": {
            "verified_facts": state.get("verified_facts") or [],
            "opinions_sentiments": state.get("opinions_sentiments") or [],
            "analytical_frameworks": state.get("analytical_frameworks") or [],
        },
        "synthesis": {
            "summary": synthesis_ctx.get("synthesis_summary", ""),
            "top_insights": synthesis_ctx.get("top_insights", []),
            "key_tensions": synthesis_ctx.get("key_tensions", []),
            "overall_confidence": synthesis_ctx.get("overall_confidence", 0.5),
            "confidence_rationale": synthesis_ctx.get("confidence_rationale", ""),
            "recommended_investigation": synthesis_ctx.get("recommended_investigation", []),
        },
        "source_data": {
            "query_agent": {
                "total_sources": total_sources,
                "stance_distribution": stance_distribution,
                "coverage_score": coverage_score,
                "top_sources": [
                    {
                        "title": s.get("title", ""),
                        "url": s.get("url", ""),
                        "trust_score": s.get("trust_score", 0),
                        "stance": s.get("stance_label", ""),
                    }
                    for s in top_sources
                ],
                "social_sentiment": social_sentiment,
            },
            "media_agent": {
                "available": media_text is not None,
                "mode": "test_data" if (media_run or {}).get("agent_name") == "media_agent" else "live",
                "summary_length": len(media_text or ""),
            },
        },
        "coordinator_trace": state.get("coordinator_trace") or [],
        "agent_errors": state.get("agent_errors") or [],
    }


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


