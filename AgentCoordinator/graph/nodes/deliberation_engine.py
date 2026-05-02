"""
DeliberationEngineNode: Multi-Perspective Deliberation Engine (Innovation 1).

Implements Hybrid Plan C from the design document:
  Phase 2.1: 4 independent perspective analyses (parallel asyncio.gather)
  Phase 2.2: Cross-examination (single LLM call with all Phase 1 results)
  Phase 2.3: Synthesis arbitration (single LLM call)

Total: 4 (parallel) + 1 + 1 = 6 LLM calls, effectively 3 sequential latency steps.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..state import CoordinatorState, DeliberationRound
from ...prompts.deliberation_prompts import (
    INDEPENDENT_ANALYSIS_PROMPT,
    CROSS_EXAMINATION_PROMPT,
    SYNTHESIS_ARBITRATION_PROMPT,
)
from ...utils.perspective_templates import get_perspectives

# Add project root to path for LLMClient
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _get_llm():
    """Lazy-load LLMClient to avoid import-time failures."""
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


def _summarize_query_agent(qa_output: Optional[Dict]) -> str:
    if not qa_output:
        return "No Query Agent data available."
    lines = [
        f"Query: {qa_output.get('original_query', '')}",
        f"Analysis type: {qa_output.get('analysis_type', '')}",
        f"Sources kept: {qa_output.get('total_sources_kept', 0)}",
        f"Coverage score: {qa_output.get('coverage_score', 0):.2f}",
        f"Stance distribution: {qa_output.get('stance_distribution', {})}",
    ]
    for cluster in qa_output.get("opinion_clusters", [])[:4]:
        lines.append(
            f"- [{cluster.get('stance', '')}] {cluster.get('core_argument', '')} "
            f"({cluster.get('source_count', 0)} sources)"
        )
    gaps = qa_output.get("knowledge_gaps", [])
    if gaps:
        lines.append(f"Knowledge gaps: {'; '.join(gaps[:3])}")
    # Social sentiment summary
    social = qa_output.get("social_sentiment")
    if social and social.get("mode") == "available":
        lines.append(f"Social CSSD: {social.get('divergence_score', 0):.3f}")
        lines.append(f"Social platforms: {social.get('platforms_queried', [])}")
        lines.append(f"Social distribution: {social.get('sentiment_distribution', {})}")
    return "\n".join(lines)


def _summarize_media_agent(media_text: Optional[str]) -> str:
    if not media_text:
        return "No Media Agent data available."
    return media_text[:2000] if len(media_text) > 2000 else media_text


def _summarize_social(qa_output: Optional[Dict]) -> str:
    if not qa_output:
        return "No social media data."
    social = qa_output.get("social_sentiment")
    if not social or social.get("mode") != "available":
        return "Social media data not available for this query."
    lines = [
        f"Mode: {social.get('mode')}",
        f"Platforms: {social.get('platforms_queried', [])}",
        f"Total posts: {social.get('total_posts', 0)}",
        f"CSSD: {social.get('divergence_score', 0):.3f}",
        f"Divergence summary: {social.get('divergence_summary', '')}",
    ]
    per_platform = social.get("per_platform", {})
    for platform, stats in per_platform.items():
        if stats and stats.get("distribution"):
            lines.append(f"{platform}: {stats['distribution']}")
    top = social.get("top_social_voices", [])
    for voice in top[:3]:
        lines.append(
            f"[{voice.get('platform')}] [{voice.get('stance')}] "
            f"{(voice.get('content', '') or '')[:100]}"
        )
    return "\n".join(lines)


async def _analyze_single_perspective(
    perspective_name: str,
    role_description: str,
    query: str,
    qa_summary: str,
    media_summary: str,
    social_summary: str,
    llm,
) -> Dict:
    """Run Phase 2.1 for a single perspective (one LLM call)."""
    prompt = INDEPENDENT_ANALYSIS_PROMPT.format(
        perspective_name=perspective_name,
        role_description=role_description,
        query=query,
        query_agent_summary=qa_summary,
        media_agent_summary=media_summary,
        social_sentiment_summary=social_summary,
    )
    try:
        response = llm.invoke(
            system_prompt=(
                "You are a structured deliberation analyst. Output ONLY valid JSON, no other text."
            ),
            user_prompt=prompt,
        )
        parsed = _parse_json_obj(response)
        if not parsed:
            parsed = {
                "perspective": perspective_name,
                "core_argument": f"[Parse error] Raw: {response[:200]}",
                "supporting_evidence": [],
                "confidence": 0.3,
                "data_gaps": [],
            }
        return parsed
    except Exception as exc:
        logger.error(f"[Deliberation] Phase 2.1 error for {perspective_name}: {exc}")
        return {
            "perspective": perspective_name,
            "core_argument": f"Analysis unavailable: {str(exc)[:100]}",
            "supporting_evidence": [],
            "confidence": 0.0,
            "data_gaps": ["LLM call failed"],
        }


async def deliberation_engine_node(state: CoordinatorState) -> dict:
    """LangGraph node: run 3-phase structured deliberation."""
    # Skip if already done (re-entry after targeted_search)
    if state.get("deliberation_rounds"):
        logger.info("[Deliberation] Already ran deliberation — supplementing with new search data")

    query = state["query"]
    analysis_type = state.get("analysis_type", "general")
    perspectives_list = get_perspectives(analysis_type)

    query_run = state.get("query_run")
    media_run = state.get("media_run")
    qa_output = query_run.get("output") if query_run else None
    media_text = media_run.get("text_output") if media_run else None

    # Incorporate any supplementary search results
    supplementary = state.get("supplementary_results") or []
    if supplementary:
        supp_text = "\n\nSUPPLEMENTARY SEARCH RESULTS (from targeted search):\n"
        for r in supplementary[:5]:
            supp_text += f"- {r.get('title', '')}: {r.get('snippet', '')[:150]}\n"
        qa_summary = _summarize_query_agent(qa_output) + supp_text
    else:
        qa_summary = _summarize_query_agent(qa_output)

    media_summary = _summarize_media_agent(media_text)
    social_summary = _summarize_social(qa_output)

    llm = _get_llm()
    rounds: List[DeliberationRound] = list(state.get("deliberation_rounds") or [])
    t0 = time.time()

    # ── Phase 2.1: Independent analyses (parallel) ──────────────────
    logger.info(f"[Deliberation] Phase 2.1: {len(perspectives_list)} independent analyses (parallel)")

    tasks = [
        _analyze_single_perspective(
            name, role, query, qa_summary, media_summary, social_summary, llm
        )
        for name, role in perspectives_list
    ]
    independent_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions from gather
    independent_analyses = []
    for i, res in enumerate(independent_results):
        if isinstance(res, Exception):
            name = perspectives_list[i][0]
            logger.error(f"[Deliberation] gather exception for {name}: {res}")
            independent_analyses.append({
                "perspective": name,
                "core_argument": f"Analysis failed: {str(res)[:100]}",
                "supporting_evidence": [],
                "confidence": 0.0,
                "data_gaps": [],
            })
        else:
            independent_analyses.append(res)

    rounds.append({
        "phase": "independent",
        "perspectives": independent_analyses,
        "consensus_points": [],
        "dissent_points": [],
        "raw_llm_output": None,
    })
    logger.info(f"[Deliberation] Phase 2.1 done in {time.time() - t0:.1f}s")

    # ── Phase 2.2: Cross-examination ────────────────────────────────
    logger.info("[Deliberation] Phase 2.2: Cross-examination")
    t1 = time.time()

    cross_result: Dict = {}
    try:
        cross_prompt = CROSS_EXAMINATION_PROMPT.format(
            query=query,
            independent_analyses_json=json.dumps(independent_analyses, ensure_ascii=False, indent=2),
        )
        cross_response = llm.invoke(
            system_prompt=(
                "You are a structured deliberation moderator. Output ONLY valid JSON."
            ),
            user_prompt=cross_prompt,
        )
        cross_result = _parse_json_obj(cross_response)
    except Exception as exc:
        logger.error(f"[Deliberation] Phase 2.2 error: {exc}")
        cross_result = {
            "cross_examination": [],
            "revised_positions": [],
            "emerging_consensus": [],
            "persistent_disagreements": [],
        }

    emerging_consensus = cross_result.get("emerging_consensus", [])
    persistent_disagreements = cross_result.get("persistent_disagreements", [])

    rounds.append({
        "phase": "cross_examination",
        "perspectives": cross_result.get("cross_examination", []),
        "consensus_points": emerging_consensus,
        "dissent_points": persistent_disagreements,
        "raw_llm_output": None,
    })
    logger.info(f"[Deliberation] Phase 2.2 done in {time.time() - t1:.1f}s")

    # ── Phase 2.3: Synthesis arbitration ────────────────────────────
    logger.info("[Deliberation] Phase 2.3: Synthesis arbitration")
    t2 = time.time()

    synthesis_result: Dict = {}
    try:
        synth_prompt = SYNTHESIS_ARBITRATION_PROMPT.format(
            query=query,
            independent_analyses_json=json.dumps(independent_analyses, ensure_ascii=False, indent=2),
            cross_examination_json=json.dumps(cross_result, ensure_ascii=False, indent=2),
        )
        synth_response = llm.invoke(
            system_prompt=(
                "You are a synthesis arbitrator. Output ONLY valid JSON."
            ),
            user_prompt=synth_prompt,
        )
        synthesis_result = _parse_json_obj(synth_response)
    except Exception as exc:
        logger.error(f"[Deliberation] Phase 2.3 error: {exc}")
        synthesis_result = {
            "synthesis_summary": f"Synthesis failed: {str(exc)[:100]}",
            "consensus_findings": [],
            "persistent_disagreements": [],
            "complementary_insights": [],
            "overall_confidence": 0.2,
            "key_unknowns": [],
        }

    all_consensus = (
        [f.get("finding", "") for f in synthesis_result.get("consensus_findings", [])]
        + emerging_consensus
    )
    all_dissents = (
        [d.get("disagreement", "") if isinstance(d, dict) else str(d)
         for d in synthesis_result.get("persistent_disagreements", [])]
        + persistent_disagreements
    )

    rounds.append({
        "phase": "synthesis_arbitration",
        "perspectives": synthesis_result.get("complementary_insights", []),
        "consensus_points": all_consensus,
        "dissent_points": all_dissents,
        "raw_llm_output": synthesis_result.get("synthesis_summary", ""),
    })
    logger.info(f"[Deliberation] Phase 2.3 done in {time.time() - t2:.1f}s")

    total_time = time.time() - t0
    trace = (
        f"[Deliberation] 3-phase complete in {total_time:.1f}s — "
        f"consensus={len(all_consensus)}, dissents={len(all_dissents)}, "
        f"confidence={synthesis_result.get('overall_confidence', 0):.2f}"
    )
    logger.info(trace)

    return {
        "deliberation_rounds": rounds,
        "deliberation_consensus": all_consensus,
        "deliberation_dissents": all_dissents,
        "coordinator_trace": [trace],
    }
