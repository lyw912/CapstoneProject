"""
SynthesisNode: MoA-style (Mixture of Agents) aggregation node.

Performs genuine synthesis — not concatenation, not selection.
Draws cross-cutting insights that emerge only when all phases are combined.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..state import CoordinatorState
from ...prompts.synthesis_prompt import SYNTHESIS_PROMPT

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _get_llm():
    from QueryEngine.llms import LLMClient
    from QueryEngine.utils.config import settings
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _parse_json_obj(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _build_deliberation_summary(state: CoordinatorState) -> str:
    """Summarize deliberation results for synthesis prompt."""
    rounds = state.get("deliberation_rounds") or []
    lines = []

    consensus = state.get("deliberation_consensus") or []
    dissents = state.get("deliberation_dissents") or []
    if consensus:
        lines.append("CONSENSUS FINDINGS:")
        for c in consensus[:5]:
            lines.append(f"  - {c}")
    if dissents:
        lines.append("PERSISTENT DISAGREEMENTS:")
        for d in dissents[:5]:
            lines.append(f"  - {d}")

    for r in rounds:
        if r.get("phase") == "synthesis_arbitration":
            summ = r.get("raw_llm_output", "")
            if summ:
                lines.append(f"\nDELIBERATION SYNTHESIS:\n{summ[:500]}")

    return "\n".join(lines) if lines else "No deliberation data."


def _build_verified_facts_summary(state: CoordinatorState) -> str:
    facts = state.get("verified_facts") or []
    if not facts:
        return "No verified facts extracted."
    lines = []
    for f in facts[:8]:
        status = f.get("verification_status", "unknown")
        conf = f.get("confidence", 0)
        lines.append(f"[{status}, {conf:.2f}] {f.get('fact', '')}")
    return "\n".join(lines)


def _build_echo_warnings_text(state: CoordinatorState) -> str:
    warnings = state.get("echo_warnings") or []
    if not warnings:
        return "No echo chamber or bias warnings detected."
    return "\n".join(f"⚠️ {w}" for w in warnings)


def _build_platform_interpretations_text(state: CoordinatorState) -> str:
    interps = state.get("platform_interpretations") or {}
    if not interps:
        return "No platform-specific data available."
    lines = []
    for platform, interp in interps.items():
        lines.append(f"[{platform}]\n{interp}\n")
    return "\n".join(lines)


def _build_divergence_hotspots_text(state: CoordinatorState) -> str:
    hotspots = state.get("divergence_hotspots") or []
    if not hotspots:
        return "No significant divergence hotspots detected."
    return "\n".join(f"- {h}" for h in hotspots)


def _assemble_synthesis_context(state: CoordinatorState, synthesis_result: Dict) -> Dict:
    """Build the full synthesis_context dict passed to ReportAgent."""
    query_run = state.get("query_run")
    qa_output = query_run.get("output") if query_run else None
    media_run = state.get("media_run")

    top_sources = []
    if qa_output:
        top_sources = sorted(
            qa_output.get("sources", []),
            key=lambda x: x.get("trust_score", 0),
            reverse=True,
        )[:15]

    return {
        "query": state["query"],
        "analysis_type": state.get("analysis_type", "general"),
        # Synthesis
        "synthesis_summary": synthesis_result.get("synthesis_summary", ""),
        "top_insights": synthesis_result.get("top_insights", []),
        "key_tensions": synthesis_result.get("key_tensions", []),
        "overall_confidence": synthesis_result.get("overall_confidence", 0.5),
        "confidence_rationale": synthesis_result.get("confidence_rationale", ""),
        "recommended_investigation": synthesis_result.get("recommended_further_investigation", []),
        # Deliberation
        "deliberation_consensus": state.get("deliberation_consensus") or [],
        "deliberation_dissents": state.get("deliberation_dissents") or [],
        "deliberation_rounds": state.get("deliberation_rounds") or [],
        # Facts and opinions
        "verified_facts": state.get("verified_facts") or [],
        "opinions_sentiments": state.get("opinions_sentiments") or [],
        "analytical_frameworks": state.get("analytical_frameworks") or [],
        # Bias and platform
        "echo_warnings": state.get("echo_warnings") or [],
        "silent_majority_hypothesis": state.get("silent_majority_hypothesis"),
        "platform_interpretations": state.get("platform_interpretations") or {},
        # Divergence
        "divergence_matrix": state.get("divergence_matrix") or {},
        "divergence_hotspots": state.get("divergence_hotspots") or [],
        # Sources
        "top_sources": top_sources,
        # Original agent outputs
        "query_agent_output": qa_output,
        "media_agent_text": (media_run.get("text_output") if media_run else None),
        # Trace
        "coordinator_trace": state.get("coordinator_trace") or [],
    }


async def synthesis_node(state: CoordinatorState) -> dict:
    """LangGraph node: MoA-style final synthesis."""
    query = state["query"]
    analysis_type = state.get("analysis_type", "general")
    t0 = time.time()

    llm = _get_llm()

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        analysis_type=analysis_type,
        deliberation_summary=_build_deliberation_summary(state),
        verified_facts_summary=_build_verified_facts_summary(state),
        echo_warnings_text=_build_echo_warnings_text(state),
        platform_interpretations_text=_build_platform_interpretations_text(state),
        divergence_hotspots_text=_build_divergence_hotspots_text(state),
    )

    synthesis_result: Dict = {}
    try:
        response = llm.invoke(
            system_prompt=(
                "You are the Synthesis Aggregator in a multi-agent analysis pipeline. "
                "Output ONLY valid JSON, no other text."
            ),
            user_prompt=prompt,
        )
        synthesis_result = _parse_json_obj(response)
    except Exception as exc:
        logger.error(f"[Synthesis] LLM call failed: {exc}")
        synthesis_result = {
            "synthesis_summary": f"Synthesis failed: {str(exc)[:200]}",
            "top_insights": [],
            "key_tensions": [],
            "overall_confidence": 0.2,
            "confidence_rationale": "LLM error",
            "recommended_further_investigation": [],
        }

    confidence = synthesis_result.get("overall_confidence", 0.5)
    synthesis_context = _assemble_synthesis_context(state, synthesis_result)

    duration = time.time() - t0
    trace = (
        f"[Synthesis] MoA aggregation complete in {duration:.1f}s — "
        f"confidence={confidence:.2f}, "
        f"insights={len(synthesis_result.get('top_insights', []))}"
    )
    logger.info(trace)

    return {
        "synthesis_context": synthesis_context,
        "synthesis_confidence": confidence,
        "coordinator_trace": [trace],
    }
