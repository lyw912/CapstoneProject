"""
OutputAssemble Node — Structured Output Assembly

Phase 1: Calculate stance_distribution only + aggregate sources.
Phase 2: Add opinion_clusters (LLM opinion clustering for each stance).
Phase 3: Add knowledge_gaps + structured_summary (LLM generated).

Phase 2 Key Changes:
  - Prioritize using classified_sources (already has stance_label + trust_score)
  - Generate OpinionCluster for each stance (via LLM call)
  - Use stance_coverage (from CoverageCheck) to calculate coverage
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from loguru import logger

from ...llms import LLMClient
from ...utils.config import settings
from ..state import OpinionCluster, QueryAgentOutput, QueryAgentState, SourceItem

# ---------------------------------------------------------------------------
# Coverage Constants (consistent with coverage_check.py)
# ---------------------------------------------------------------------------

_STANCE_THRESHOLDS: Dict[str, int] = {
    "support":  2,
    "oppose":   2,
    "official": 1,
    "neutral":  1,
}


def _compute_coverage_score(stance_counts: Dict[str, int]) -> float:
    """Calculate coverage score (0–1) based on STANCE_THRESHOLDS."""
    total_required = sum(_STANCE_THRESHOLDS.values())
    total_met = sum(
        min(stance_counts.get(s, 0), c)
        for s, c in _STANCE_THRESHOLDS.items()
    )
    return round(total_met / max(total_required, 1), 3)


# ---------------------------------------------------------------------------
# LLM Tools
# ---------------------------------------------------------------------------

OPINION_CLUSTER_PROMPT = """You are a public opinion analyst. For the topic "{query}", here is content with "{stance}" stance:

{sources_text}

Please summarize:
1. Core argument (1 sentence, concisely summarize the main claim of this stance)
2. Most representative original quote (within 100 characters, can be directly excerpted)

Output only JSON in the format:
{{"core_argument": "...", "representative_quote": "..."}}"""


def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _parse_json_obj(text: str) -> dict:
    """Extract JSON object from LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _build_sources_text(srcs: List[dict], max_sources: int = 5) -> str:
    """Convert source list to text for LLM reading (at most max_sources items)."""
    lines = []
    for s in srcs[:max_sources]:
        title = s.get("title", "(No title)")
        url = s.get("url", "")
        snippet = (s.get("snippet") or "")[:200]
        lines.append(f"- [{title}]({url}): {snippet}")
    return "\n".join(lines) if lines else "(No sources)"


KNOWLEDGE_GAPS_PROMPT = """Based on the following public opinion analysis results, identify 3-5 still missing information dimensions:

Topic: {query}
Covered stances: {covered_stances}
Missing stances: {missing_stances}
Source platform distribution: {platforms}

Please list questions we still don't know, in the format "We still don't know...".
Output only a JSON array: ["We still don't know...", "We still don't know..."]"""


async def _identify_knowledge_gaps(
    query: str,
    stance_coverage: Dict[str, int],
    missing_stances: List[str],
    sources: List[dict],
    llm: LLMClient,
) -> List[str]:
    """Identify knowledge gaps (3-5 items)"""
    if not sources:
        return ["We still don't know any information about this topic"]

    covered = list(stance_coverage.keys())
    platforms = list(set(s.get("platform", "") for s in sources if s.get("platform")))[:10]

    prompt = KNOWLEDGE_GAPS_PROMPT.format(
        query=query,
        covered_stances=", ".join(covered) if covered else "None",
        missing_stances=", ".join(missing_stances) if missing_stances else "None",
        platforms=", ".join(platforms) if platforms else "None",
    )

    try:
        response = llm.invoke(
            system_prompt="You are a public opinion analysis expert. Output only a JSON array, no other text.",
            user_prompt=prompt,
        )
        gaps = _parse_json_array(response)
        return gaps[:5] if gaps else []
    except Exception as exc:
        logger.warning(f"[OutputAssemble] Knowledge gap identification failed: {exc}")
        return []


def _parse_json_array(text: str) -> list:
    """Extract JSON array from LLM response"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


async def _generate_opinion_cluster(
    stance: str,
    srcs: List[dict],
    query: str,
    total_sources: int,
    llm: LLMClient,
) -> Optional[OpinionCluster]:
    """
    Call LLM to generate OpinionCluster for a single stance.
    If LLM call fails, return a rule-based minimal cluster.
    """
    sources_text = _build_sources_text(srcs)
    prompt = OPINION_CLUSTER_PROMPT.format(
        query=query,
        stance=stance,
        sources_text=sources_text,
    )

    core_argument = ""
    representative_quote = ""

    try:
        response = llm.invoke(
            system_prompt="You are a public opinion analysis expert. Output only JSON, no other text.",
            user_prompt=prompt,
        )
        parsed = _parse_json_obj(response)
        core_argument = parsed.get("core_argument", "")
        representative_quote = parsed.get("representative_quote", "")
    except Exception as exc:
        logger.warning(f"[OutputAssemble] Stance '{stance}' clustering LLM failed: {exc}")
        # Fallback: Use the first source's snippet as representative quote
        if srcs:
            core_argument = f"Sources with {stance} stance (total {len(srcs)} items)"
            representative_quote = (srcs[0].get("snippet") or "")[:100]

    if not core_argument:
        return None

    return {
        "cluster_id":            f"cluster_{stance}",
        "stance":                stance,
        "core_argument":         core_argument,
        "evidence_sources":      [s["source_id"] for s in srcs],
        "representative_quote":  representative_quote,
        "estimated_proportion":  round(len(srcs) / max(total_sources, 1), 3),
        "source_count":          len(srcs),
    }


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------

async def output_assemble_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: Assemble final structured output.

    Phase 2:
    - Prioritize using classified_sources (has stance_label + trust_score)
    - Call LLM to generate opinion_cluster for each stance
    - knowledge_gaps / structured_summary reserved for Phase 3 completion
    """
    # ------------------------------------------------------------------
    # 1. Get sources (priority: classified > scored > deduped > raw)
    # ------------------------------------------------------------------
    sources: List[SourceItem] = (
        state.get("classified_sources")
        or state.get("scored_sources")
        or state.get("deduped_sources")
        or state.get("raw_sources")
        or []
    )

    query = state.get("original_query", "")
    total_raw = len(state.get("raw_sources") or [])
    total_kept = len(sources)

    # ------------------------------------------------------------------
    # 2. Stance distribution (Phase 2: from actual stance_label)
    # ------------------------------------------------------------------
    stance_counts = Counter(
        s.get("stance_label") or "unclassified" for s in sources
    )
    total = max(total_kept, 1)
    stance_distribution: Dict[str, float] = {
        stance: round(count / total, 3)
        for stance, count in stance_counts.items()
    }

    # ------------------------------------------------------------------
    # 3. Coverage score
    #    Prioritize using stance_coverage calculated by CoverageCheck; otherwise calculate yourself
    # ------------------------------------------------------------------
    coverage_counts = state.get("stance_coverage") or dict(stance_counts)
    coverage_score = _compute_coverage_score(coverage_counts)

    # ------------------------------------------------------------------
    # 4. Source sorting (by trust_score descending)
    # ------------------------------------------------------------------
    sorted_sources = sorted(
        sources,
        key=lambda x: x.get("trust_score", 0.0),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # 5. Phase 2: OpinionCluster generation (one cluster per stance)
    # ------------------------------------------------------------------
    opinion_clusters: List[OpinionCluster] = []

    # Only generate clusters for stances with actual sources (exclude unclassified)
    stance_groups: Dict[str, List[dict]] = defaultdict(list)
    for s in sources:
        label = s.get("stance_label") or "unclassified"
        if label != "unclassified":
            stance_groups[label].append(s)

    if stance_groups:
        llm = _get_llm_client()
        # Generate in descending order of stance frequency (ensure most important stances are processed first)
        sorted_stances = sorted(
            stance_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        for stance, srcs in sorted_stances:
            cluster = await _generate_opinion_cluster(
                stance=stance,
                srcs=srcs,
                query=query,
                total_sources=total_kept,
                llm=llm,
            )
            if cluster:
                opinion_clusters.append(cluster)

    # ------------------------------------------------------------------
    # 6. Knowledge gap identification
    # ------------------------------------------------------------------
    knowledge_gaps = await _identify_knowledge_gaps(
        query=query,
        stance_coverage=coverage_counts,
        missing_stances=state.get("missing_stances") or [],
        sources=sources,
        llm=llm if stance_groups else _get_llm_client(),
    )

    # ------------------------------------------------------------------
    # 7. Assemble QueryAgentOutput
    # ------------------------------------------------------------------
    output: QueryAgentOutput = {
        "original_query":     query,
        "analysis_type":      state.get("analysis_type", "general"),
        "search_iterations":  state.get("search_iterations", 0),
        "total_sources_found": total_raw,
        "total_sources_kept":  total_kept,
        "stance_distribution": stance_distribution,
        "opinion_clusters":    opinion_clusters,
        "sources":             sorted_sources,
        "knowledge_gaps":      knowledge_gaps,
        "coverage_score":      coverage_score,
        "structured_summary":  "",            # Phase 3 optional
        "social_sentiment":    state.get("social_sentiment"),
        "trace_log":           state.get("trace_log") or [],
    }

    trace = (
        f"[OutputAssemble] Sources={total_kept}/{total_raw}, "
        f"Stance distribution={stance_distribution}, "
        f"SCS={coverage_score:.2f}, "
        f"clusters={len(opinion_clusters)}, "
        f"knowledge gaps={len(knowledge_gaps)}, "
        f"social_mode={state.get('mindspider_mode', 'disabled')}"
    )
    logger.info(trace)

    return {
        "query_agent_output": output,
        "trace_log": [trace],
    }
