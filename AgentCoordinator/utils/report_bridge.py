"""
Report bridge: converts AgentCoordinator artifacts into ReportEngine inputs.

The bridge keeps the Coordinator/ReportEngine boundary explicit:
- Coordinator owns evidence collection, deliberation, and structured synthesis.
- ReportEngine owns template/layout/chapter rendering.

Raw Chinese source text may appear as quoted evidence. Bridge-generated labels,
instructions, and report scaffolding are English.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


ENGLISH_OUTPUT_CONSTRAINT = (
    "Write all generated report prose, headings, captions, table labels, and "
    "explanatory text in English only. Do not output Chinese characters. "
    "Translate Chinese source material into English while preserving facts. "
    "Proper nouns and URLs may remain unchanged."
)


def build_report_prompt(synthesis_context: Dict) -> str:
    """
    Convert synthesis_context into a comprehensive markdown prompt
    that ReportAgent can use to generate the final report.
    """
    query = synthesis_context.get("query", "")
    facts = synthesis_context.get("verified_facts", [])
    opinions = synthesis_context.get("opinions_sentiments", [])
    frameworks = synthesis_context.get("analytical_frameworks", [])
    consensus = synthesis_context.get("deliberation_consensus", [])
    dissents = synthesis_context.get("deliberation_dissents", [])
    echo_warnings = synthesis_context.get("echo_warnings", [])
    platform_interps = synthesis_context.get("platform_interpretations", {})
    sources = synthesis_context.get("top_sources", [])

    parts = [f"# Multi-Agent Analysis Report: {query}", ""]
    parts.append(f"> Language rule: {ENGLISH_OUTPUT_CONSTRAINT}")
    parts.append("")

    if facts:
        parts.extend(["## Verified Facts", ""])
        for fact in facts[:10]:
            status = fact.get("verification_status", "unknown")
            conf = fact.get("confidence", 0)
            parts.append(f"- **[{status}, conf={conf:.2f}]** {fact.get('fact', '')}")
        parts.append("")

    if opinions:
        parts.extend(["## Public Opinions & Sentiments", ""])
        for opinion in opinions[:8]:
            parts.append(f"- **{opinion.get('perspective', '')}** - held by: {opinion.get('holders', '')}")
        parts.append("")

    if consensus:
        parts.extend(["## Cross-Perspective Consensus", ""])
        for point in consensus:
            parts.append(f"- {point}")
        parts.append("")

    if dissents:
        parts.extend(["## Persistent Disagreements", ""])
        for point in dissents:
            parts.append(f"- {point}")
        parts.append("")

    if platform_interps:
        parts.extend(["## Platform-Specific Interpretations", ""])
        for platform, interp in platform_interps.items():
            parts.extend([f"### {platform}", str(interp), ""])

    if echo_warnings:
        parts.extend(["## Bias & Limitations", ""])
        for warning in echo_warnings:
            parts.append(f"Warning: {warning}")
        parts.append("")

    if frameworks:
        parts.extend(["## Analytical Frameworks", ""])
        for framework in frameworks[:5]:
            fw_type = framework.get("framework", "")
            analysis = framework.get("analysis", "")
            certainty = framework.get("certainty", "")
            parts.append(f"**{fw_type}** [{certainty}]: {analysis}")

    if sources:
        parts.extend(["## Key Sources", ""])
        for source in sources[:15]:
            title = source.get("title", "(no title)")
            url = source.get("url", "")
            ts = source.get("trust_score", 0)
            parts.append(f"- [{title}]({url}) - trust: {ts:.2f}")
        parts.append("")

    return "\n".join(parts)


def synthesis_context_to_markdown(synthesis_context: Dict) -> str:
    """Simple markdown fallback when ReportEngine is unavailable."""
    return build_report_prompt(synthesis_context)


def coordinator_output_to_report_engine_inputs(coordinator_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert clean coordinator_output.json into ReportAgent.generate_report inputs.

    ReportEngine currently expects a QueryEngine text report, a MediaEngine text
    report, and optional forum logs. Coordinator output is richer than that, so
    this adapter serializes it into two evidence-dense English Markdown inputs
    plus a compact trace log. The original structured JSON remains embedded in
    the Query report for lossless downstream use.
    """
    query = str(coordinator_output.get("query", ""))
    query_report = _build_query_engine_report(coordinator_output)
    media_report = _build_media_engine_report(coordinator_output)
    forum_logs = _build_forum_logs(coordinator_output)

    return {
        "query": query,
        "reports": [query_report, media_report],
        "forum_logs": forum_logs,
        "custom_template": build_english_report_template(),
        "metadata": {
            "schema_version": coordinator_output.get("schema_version", "1.0"),
            "analysis_type": coordinator_output.get("analysis_type", "general"),
            "generated_at": coordinator_output.get("generated_at", ""),
            "language_constraint": ENGLISH_OUTPUT_CONSTRAINT,
        },
    }


def generate_report_engine_html(
    coordinator_output: Dict[str, Any],
    report_agent: Optional[Any] = None,
    *,
    save_report: bool = True,
    stream_handler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run ReportEngine from a Coordinator output artifact.

    A ReportAgent instance may be injected for tests. If omitted, the adapter
    imports and creates the standard ReportEngine agent lazily.
    """
    inputs = coordinator_output_to_report_engine_inputs(coordinator_output)
    if report_agent is None:
        from ReportEngine import create_agent

        report_agent = create_agent()

    result = report_agent.generate_report(
        query=inputs["query"],
        reports=inputs["reports"],
        forum_logs=inputs["forum_logs"],
        custom_template=inputs["custom_template"],
        save_report=save_report,
        stream_handler=stream_handler,
    )
    if isinstance(result, dict):
        result.setdefault("adapter_metadata", inputs["metadata"])
    return result


def build_english_report_template() -> str:
    """Return a compact template tuned for Coordinator-sourced reports."""
    return """# Multi-Source Public Opinion Analysis Report

## 1. Executive Summary
- Summarize the overall finding, confidence level, and strongest evidence.
- State the language rule: generated prose must be English; original Chinese may only appear as verbatim evidence.

## 2. Data and Methodology
- Explain web/news retrieval, social-media evidence, deliberation, CSSD, SCS, and TrustScore.
- Include limitations and data coverage.

## 3. Evidence Overview
- Present source counts, stance distribution, representative sources, and social-media voices.
- Preserve traceability to URLs and platforms.

## 4. Cross-Source Divergence and Deliberation
- Explain CSSD hotspots, consensus points, and persistent disagreements.
- Distinguish verified facts from opinions.

## 5. Interpretation, Risks, and Recommendations
- Provide platform-aware interpretation, key tensions, bias warnings, and recommended follow-up work.

## Appendix
- Include source list, raw metric tables, and coordinator trace where useful.
"""


def _build_query_engine_report(coordinator_output: Dict[str, Any]) -> str:
    query = coordinator_output.get("query", "")
    synthesis = coordinator_output.get("synthesis", {}) or {}
    source_data = coordinator_output.get("source_data", {}) or {}
    qa = source_data.get("query_agent", {}) or {}
    divergence = coordinator_output.get("divergence_matrix", {}) or {}
    fact_opinion = coordinator_output.get("fact_opinion_separation", {}) or {}
    bias = coordinator_output.get("bias_analysis", {}) or {}

    lines: List[str] = [
        f"# Query Agent Evidence Package: {query}",
        "",
        f"Language rule: {ENGLISH_OUTPUT_CONSTRAINT}",
        "",
        "## Synthesis",
        str(synthesis.get("summary", "")),
        "",
        "## Retrieval Metrics",
        f"- Total retained sources: {qa.get('total_sources', 0)}",
        f"- Stance coverage score: {qa.get('coverage_score', 0)}",
        f"- CSSD pair count: {len(divergence.get('pairs', {}) or {})}",
        f"- Maximum CSSD: {(divergence.get('max_divergence') or {}).get('pair', 'N/A')} = {(divergence.get('max_divergence') or {}).get('value', 0)}",
        "",
    ]

    stance_dist = qa.get("stance_distribution", {}) or {}
    if stance_dist:
        lines.extend(["## Stance Distribution", ""])
        for stance, value in sorted(stance_dist.items(), key=lambda item: -float(item[1])):
            lines.append(f"- {stance}: {float(value):.1%}")
        lines.append("")

    top_sources = qa.get("top_sources", []) or []
    if top_sources:
        lines.extend(["## Top Sources", ""])
        for source in top_sources:
            lines.append(
                f"- [{source.get('title', '(untitled)')}]({source.get('url', '')}) "
                f"| stance={source.get('stance', '')} | trust={float(source.get('trust_score', 0) or 0):.2f}"
            )
        lines.append("")

    verified_facts = fact_opinion.get("verified_facts", []) or []
    if verified_facts:
        lines.extend(["## Verified Facts", ""])
        for fact in verified_facts:
            lines.append(
                f"- [{fact.get('verification_status', 'unknown')}, "
                f"confidence={float(fact.get('confidence', 0) or 0):.2f}] {fact.get('fact', '')}"
            )
        lines.append("")

    warnings = bias.get("echo_warnings", []) or []
    if warnings:
        lines.extend(["## Bias Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Structured Coordinator Output",
            "```json",
            json.dumps(coordinator_output, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines)


def _build_media_engine_report(coordinator_output: Dict[str, Any]) -> str:
    query = coordinator_output.get("query", "")
    platform_interps = coordinator_output.get("platform_interpretations", {}) or {}
    source_data = coordinator_output.get("source_data", {}) or {}
    qa = source_data.get("query_agent", {}) or {}
    social = qa.get("social_sentiment") or {}
    media = source_data.get("media_agent", {}) or {}

    lines: List[str] = [
        f"# Media and Social Evidence Package: {query}",
        "",
        f"Language rule: {ENGLISH_OUTPUT_CONSTRAINT}",
        "",
        "## Media Agent Availability",
        f"- Available: {bool(media.get('available'))}",
        f"- Mode: {media.get('mode', 'unknown')}",
        f"- Summary length: {media.get('summary_length', 0)}",
        "",
    ]

    if platform_interps:
        lines.extend(["## Platform-Aware Interpretations", ""])
        for platform, interpretation in platform_interps.items():
            if interpretation:
                lines.extend([f"### {platform}", str(interpretation), ""])

    if social:
        lines.extend(
            [
                "## Social Media Metrics",
                f"- Mode: {social.get('mode', 'unknown')}",
                f"- Platforms queried: {', '.join(social.get('platforms_queried', []) or [])}",
                f"- Total posts: {social.get('total_posts', 0)}",
                f"- Total comments: {social.get('total_comments', 0)}",
                f"- Web/social CSSD: {social.get('divergence_score', 0)}",
                "",
            ]
        )
        social_dist = social.get("sentiment_distribution", {}) or {}
        if social_dist:
            lines.extend(["### Social Stance Distribution", ""])
            for stance, value in sorted(social_dist.items(), key=lambda item: -float(item[1])):
                lines.append(f"- {stance}: {float(value):.1%}")
            lines.append("")

        voices = social.get("top_social_voices", []) or []
        if voices:
            lines.extend(["### Representative Social Voices", "Original-language posts are verbatim evidence.", ""])
            for voice in voices[:10]:
                lines.append(
                    f"- [{voice.get('platform', '')}] stance={voice.get('stance', '')} "
                    f"time={voice.get('publish_time', '')} url={voice.get('url', '')}"
                )
                lines.append(f"  > {(voice.get('content', '') or '')[:260]}")
            lines.append("")

    return "\n".join(lines)


def _build_forum_logs(coordinator_output: Dict[str, Any]) -> str:
    deliberation = coordinator_output.get("deliberation", {}) or {}
    trace = coordinator_output.get("coordinator_trace", []) or []
    lines: List[str] = [
        "[Coordinator Bridge] ReportEngine handoff generated from structured Coordinator output.",
        f"[Coordinator Bridge] {ENGLISH_OUTPUT_CONSTRAINT}",
        "",
        "[Deliberation Consensus]",
    ]
    for point in deliberation.get("final_consensus", []) or []:
        lines.append(f"- {point}")
    lines.append("")
    lines.append("[Persistent Disagreements]")
    for point in deliberation.get("final_dissents", []) or []:
        lines.append(f"- {point}")
    if trace:
        lines.extend(["", "[Coordinator Trace]"])
        lines.extend(str(entry) for entry in trace)
    return "\n".join(lines)
