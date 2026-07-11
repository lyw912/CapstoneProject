"""
CoordinatorOutputSchema: compatibility JSON artifact documentation.

This module specifies the exact structure of coordinator_output.json so that
ReportAgent knows precisely what fields to consume. Every field is documented
with its type and semantics.

Active runtime note:
    /api/coordinator/run now uses AgentCoordinator's internal intelligence layer.
    The active artifact schema is "2.1-coordinator-intelligence" and includes the
    top-level "coordinator_intelligence" evidence ledger. Compatibility fields
    such as source_data, synthesis, divergence_matrix, and deliberation are views
    over that ledger. The builder below remains for compatibility graph/report bridge
    tests and older call sites.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Field-level documentation dictionary (used by ReportAgent for validation)
# ---------------------------------------------------------------------------

COORDINATOR_OUTPUT_SCHEMA: Dict[str, Any] = {
    "schema_version": {
        "type": "str",
        "description": "Schema version string, e.g. '1.0'. Bump when breaking changes are made.",
    },
    "query": {
        "type": "str",
        "description": "The original user query that was analyzed.",
    },
    "analysis_type": {
        "type": "str",
        "description": "Query classification: event|brand|policy|technology|general.",
    },
    "generated_at": {
        "type": "str (ISO 8601)",
        "description": "UTC timestamp when this output was generated.",
    },
    "pipeline_duration_seconds": {
        "type": "float",
        "description": "Total wall-clock time for the full coordinator pipeline.",
    },
    "coordinator_intelligence": {
        "type": "dict | optional",
        "description": "Internal CoordinatorIntelligenceArtifact ledger when schema_version is 2.1-coordinator-intelligence.",
    },
    # -------------------------------------------------------------------
    "divergence_matrix": {
        "type": "dict",
        "description": "Cross-Source Sentiment Divergence (CSSD) results.",
        "fields": {
            "pairs": {
                "type": "dict[str, float]",
                "description": (
                    "Map of 'source_a|source_b' to CSSD delta value (0.0-1.0). "
                    "Larger values indicate stronger divergence between those sources."
                ),
            },
            "hotspots": {
                "type": "list[str]",
                "description": "Human-readable descriptions of pairs with CSSD > 0.3.",
            },
            "max_divergence": {
                "type": "dict[str, Any]",
                "description": "Pair with highest CSSD: {'pair': str, 'value': float}.",
            },
            "min_divergence": {
                "type": "dict[str, Any]",
                "description": "Pair with lowest CSSD: {'pair': str, 'value': float}.",
            },
        },
    },
    # -------------------------------------------------------------------
    "deliberation": {
        "type": "dict",
        "description": "Multi-Perspective Deliberation results (Innovation 1).",
        "fields": {
            "analysis_type": {"type": "str", "description": "Same as top-level analysis_type."},
            "perspectives_used": {
                "type": "list[str]",
                "description": "Names of the analytical perspectives used (e.g. ['Facts & Data', ...]).",
            },
            "phases": {
                "type": "list[dict]",
                "description": (
                    "Ordered list of deliberation phases. Each phase dict has: "
                    "phase (str), summary (str), consensus_points (list[str]), dissent_points (list[str])."
                ),
            },
            "final_consensus": {
                "type": "list[str]",
                "description": "Cross-perspective consensus points from the final synthesis phase.",
            },
            "final_dissents": {
                "type": "list[str]",
                "description": "Persisting disagreements that were not resolved.",
            },
            "confidence": {
                "type": "float (0-1)",
                "description": "Deliberation confidence level aggregated from all phases.",
            },
        },
    },
    # -------------------------------------------------------------------
    "gap_filling": {
        "type": "dict",
        "description": "CRAG-driven gap filling results (Innovation 2).",
        "fields": {
            "rounds_performed": {
                "type": "int",
                "description": "Number of targeted-search rounds actually executed (0 = skipped).",
            },
            "gaps_detected": {
                "type": "list[dict]",
                "description": "Each entry: {'description': str, 'source': str ('tavily'|'mindspider_db')}.",
            },
            "results_found": {
                "type": "int",
                "description": "Total number of supplementary results retrieved across all rounds.",
            },
        },
    },
    # -------------------------------------------------------------------
    "platform_interpretations": {
        "type": "dict[str, str | None]",
        "description": (
            "Platform-Aware Interpretation results (Innovation 3). "
            "Keys are platform names (weibo, zhihu, bilibili, web, etc.). "
            "Values are multi-sentence interpretation strings or None if no data."
        ),
    },
    # -------------------------------------------------------------------
    "bias_analysis": {
        "type": "dict",
        "description": "Echo chamber detection results (Innovation 4).",
        "fields": {
            "echo_warnings": {
                "type": "list[str]",
                "description": "Human-readable warnings about detected echo chambers or bias clusters.",
            },
            "silent_majority_hypothesis": {
                "type": "str | None",
                "description": "If detected, a hypothesis about what the silent majority may believe.",
            },
        },
    },
    "fact_opinion_separation": {
        "type": "dict",
        "description": "Fact-Opinion separation results (Innovation 4, second half).",
        "fields": {
            "verified_facts": {
                "type": "list[dict]",
                "description": (
                    "Each entry: {'fact': str, 'sources': list[str], "
                    "'verification_status': str, 'confidence': float}."
                ),
            },
            "opinions_sentiments": {
                "type": "list[dict]",
                "description": (
                    "Each entry: {'perspective': str, 'holders': str, "
                    "'sentiment_intensity': str, 'potential_biases': list[str]}."
                ),
            },
            "analytical_frameworks": {
                "type": "list[dict]",
                "description": (
                    "Each entry: {'framework': str, 'analysis': str, 'certainty': str}."
                ),
            },
        },
    },
    # -------------------------------------------------------------------
    "synthesis": {
        "type": "dict",
        "description": "MoA (Mixture of Agents) final synthesis output.",
        "fields": {
            "summary": {
                "type": "str",
                "description": "High-level synthesis narrative.",
            },
            "top_insights": {
                "type": "list[dict]",
                "description": (
                    "Each entry: {'insight': str, 'basis': str, 'confidence': float}."
                ),
            },
            "key_tensions": {
                "type": "list[dict]",
                "description": (
                    "Each entry: {'tension': str, 'between': list[str], 'significance': str}."
                ),
            },
            "overall_confidence": {
                "type": "float (0-1)",
                "description": "Pipeline-level confidence in the synthesis conclusions.",
            },
            "recommended_investigation": {
                "type": "list[str]",
                "description": "Suggested follow-up investigation items.",
            },
        },
    },
    # -------------------------------------------------------------------
    "source_data": {
        "type": "dict",
        "description": "Summary of raw data obtained from QueryAgent and MediaAgent.",
        "fields": {
            "query_agent": {
                "type": "dict",
                "description": "Stats from QueryAgent run.",
                "fields": {
                    "total_sources": {"type": "int", "description": "Number of sources retrieved."},
                    "stance_distribution": {
                        "type": "dict[str, float]",
                        "description": "Fraction of sources per stance label.",
                    },
                    "coverage_score": {
                        "type": "float (0-1)",
                        "description": "Estimated topic coverage score.",
                    },
                    "top_sources": {
                        "type": "list[dict]",
                        "description": (
                            "Top sources by trust_score: "
                            "{'title': str, 'url': str, 'trust_score': float, 'stance': str}."
                        ),
                    },
                    "social_sentiment": {
                        "type": "dict | None",
                        "description": "Aggregated social sentiment data if available.",
                    },
                },
            },
            "media_agent": {
                "type": "dict",
                "description": "Summary of MediaAgent contribution.",
                "fields": {
                    "available": {"type": "bool", "description": "Whether MediaAgent ran successfully."},
                    "mode": {
                        "type": "str ('live' | 'test_data')",
                        "description": "Whether live crawl or test fixture data was used.",
                    },
                    "summary_length": {
                        "type": "int",
                        "description": "Character length of MediaAgent text output.",
                    },
                },
            },
        },
    },
    # -------------------------------------------------------------------
    "coordinator_trace": {
        "type": "list[str]",
        "description": "Ordered list of trace log entries from each pipeline node.",
    },
    "agent_errors": {
        "type": "list[str]",
        "description": "Any errors raised by individual nodes (non-fatal). Empty list if all clean.",
    },
}


# ---------------------------------------------------------------------------
# Builder function: extracts clean JSON from coordinator run result
# ---------------------------------------------------------------------------

def build_coordinator_output(
    result: Dict[str, Any],
    query: str,
    duration_seconds: float,
) -> Dict[str, Any]:
    """
    Build the clean coordinator_output.json artifact from a coordinator run result.

    Args:
        result: The dict returned by AgentCoordinator.run().
        query: The original query string.
        duration_seconds: Total pipeline duration in seconds.

    Returns:
        A clean structured dict conforming to COORDINATOR_OUTPUT_SCHEMA.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Divergence matrix
    raw_matrix = result.get("divergence_matrix") or {}
    hotspots = result.get("divergence_hotspots") or []

    # raw_matrix may be {(a,b): float} or {"a|b": float}; normalize to str keys.
    pairs: Dict[str, float] = {}
    for k, v in raw_matrix.items():
        if isinstance(k, tuple):
            pairs["|".join(k)] = float(v)
        else:
            pairs[str(k)] = float(v)

    max_pair = {"pair": "", "value": 0.0}
    min_pair = {"pair": "", "value": 1.0}
    for pair_key, delta in pairs.items():
        if delta >= max_pair["value"]:
            max_pair = {"pair": pair_key, "value": delta}
        if delta <= min_pair["value"]:
            min_pair = {"pair": pair_key, "value": delta}
    if not pairs:
        max_pair = {"pair": "N/A", "value": 0.0}
        min_pair = {"pair": "N/A", "value": 0.0}

    divergence_matrix_out = {
        "pairs": pairs,
        "hotspots": hotspots,
        "max_divergence": max_pair,
        "min_divergence": min_pair,
    }

    # Deliberation
    synthesis_ctx = result.get("synthesis_context") or {}
    delib_rounds_raw = synthesis_ctx.get("deliberation_rounds") or []

    perspectives_used: List[str] = []
    phases_out: List[Dict] = []
    for rd in delib_rounds_raw:
        phase_name = rd.get("phase", "unknown")
        persp_list = rd.get("perspectives") or []
        if phase_name == "independent":
            for p in persp_list:
                pname = p.get("perspective", "")
                if pname and pname not in perspectives_used:
                    perspectives_used.append(pname)
        summary_text = rd.get("raw_llm_output") or ""
        phases_out.append(
            {
                "phase": phase_name,
                "summary": summary_text[:500] if summary_text else "",
                "consensus_points": rd.get("consensus_points") or [],
                "dissent_points": rd.get("dissent_points") or [],
            }
        )

    final_consensus = result.get("deliberation_consensus") or []
    final_dissents = result.get("deliberation_dissents") or []

    deliberation_out = {
        "analysis_type": synthesis_ctx.get("analysis_type", "general"),
        "perspectives_used": perspectives_used,
        "phases": phases_out,
        "final_consensus": final_consensus,
        "final_dissents": final_dissents,
        "confidence": result.get("synthesis_confidence") or synthesis_ctx.get("overall_confidence", 0.5),
    }

    # Gap filling
    search_gaps = synthesis_ctx.get("search_gaps") or result.get("search_gaps") or []
    supplementary = synthesis_ctx.get("supplementary_results") or result.get("supplementary_results") or []
    search_rounds = synthesis_ctx.get("search_rounds") or result.get("search_rounds") or 0

    gaps_detected_out = []
    for g in search_gaps:
        gaps_detected_out.append(
            {
                "description": g.get("description", ""),
                "source": g.get("target_source", "tavily"),
            }
        )

    gap_filling_out = {
        "rounds_performed": search_rounds,
        "gaps_detected": gaps_detected_out,
        "results_found": len(supplementary),
    }

    # Platform interpretations
    platform_interps = result.get("platform_interpretations") or synthesis_ctx.get("platform_interpretations") or {}
    # Ensure None values are preserved as None (not absent)
    platform_interps_out: Dict[str, Optional[str]] = dict(platform_interps)

    # Bias analysis
    echo_warnings = result.get("echo_warnings") or synthesis_ctx.get("echo_warnings") or []
    silent_majority = synthesis_ctx.get("silent_majority_hypothesis")

    bias_analysis_out = {
        "echo_warnings": echo_warnings,
        "silent_majority_hypothesis": silent_majority,
    }

    # Fact-opinion separation
    verified_facts = result.get("verified_facts") or synthesis_ctx.get("verified_facts") or []
    opinions_sentiments = synthesis_ctx.get("opinions_sentiments") or []
    analytical_frameworks = synthesis_ctx.get("analytical_frameworks") or []

    fact_opinion_out = {
        "verified_facts": verified_facts,
        "opinions_sentiments": opinions_sentiments,
        "analytical_frameworks": analytical_frameworks,
    }

    # Synthesis
    synthesis_out = {
        "summary": synthesis_ctx.get("synthesis_summary", ""),
        "top_insights": synthesis_ctx.get("top_insights") or [],
        "key_tensions": synthesis_ctx.get("key_tensions") or [],
        "overall_confidence": synthesis_ctx.get("overall_confidence", 0.5),
        "recommended_investigation": synthesis_ctx.get("recommended_investigation") or [],
    }

    # Source data
    qa_output = synthesis_ctx.get("query_agent_output") or {}
    top_sources_raw = synthesis_ctx.get("top_sources") or []
    top_sources_out = []
    for s in top_sources_raw[:10]:
        top_sources_out.append(
            {
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "trust_score": float(s.get("trust_score", 0.0)),
                "stance": s.get("stance_label") or s.get("stance", ""),
            }
        )

    stance_dist = {}
    if qa_output:
        raw_dist = qa_output.get("stance_distribution") or {}
        for st, cnt in raw_dist.items():
            stance_dist[str(st)] = float(cnt)

    coverage_score = 0.0
    if qa_output:
        coverage_score = float(qa_output.get("coverage_score", 0.0))

    social_sentiment = None
    if qa_output:
        social_sentiment = qa_output.get("social_sentiment")

    media_text = synthesis_ctx.get("media_agent_text") or ""
    # Detect whether media agent used live data or test fixture
    media_mode = "live"
    if "[INJECTED TEST DATA]" in (media_text or "") or "[TEST DATA]" in (media_text or ""):
        media_mode = "test_data"

    source_data_out = {
        "query_agent": {
            "total_sources": len(top_sources_raw),
            "stance_distribution": stance_dist,
            "coverage_score": coverage_score,
            "top_sources": top_sources_out,
            "social_sentiment": social_sentiment,
        },
        "media_agent": {
            "available": bool(media_text),
            "mode": media_mode,
            "summary_length": len(media_text) if media_text else 0,
        },
    }

    # Assemble final output
    output = {
        "schema_version": "1.0",
        "query": query,
        "analysis_type": synthesis_ctx.get("analysis_type", "general"),
        "generated_at": now,
        "pipeline_duration_seconds": round(duration_seconds, 2),
        "divergence_matrix": divergence_matrix_out,
        "deliberation": deliberation_out,
        "gap_filling": gap_filling_out,
        "platform_interpretations": platform_interps_out,
        "bias_analysis": bias_analysis_out,
        "fact_opinion_separation": fact_opinion_out,
        "synthesis": synthesis_out,
        "source_data": source_data_out,
        "coordinator_trace": result.get("coordinator_trace") or [],
        "agent_errors": result.get("agent_errors") or [],
    }

    return output
