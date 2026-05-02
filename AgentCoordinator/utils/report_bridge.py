"""
Report bridge: converts CoordinatorState's synthesis_context into a format
suitable for ReportEngine.generate().

Also provides a text-only fallback when ReportEngine is unavailable.
"""

from __future__ import annotations

import json
from typing import Dict, Optional


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
    divergence_matrix = synthesis_context.get("divergence_matrix", {})
    sources = synthesis_context.get("top_sources", [])

    parts = [f"# Multi-Agent Analysis Report: {query}\n"]

    if facts:
        parts.append("## Verified Facts\n")
        for f in facts[:10]:
            status = f.get("verification_status", "unknown")
            conf = f.get("confidence", 0)
            parts.append(f"- **[{status}, conf={conf:.2f}]** {f.get('fact', '')}")
        parts.append("")

    if opinions:
        parts.append("## Public Opinions & Sentiments\n")
        for op in opinions[:8]:
            parts.append(f"- **{op.get('perspective', '')}** — held by: {op.get('holders', '')}")
        parts.append("")

    if consensus:
        parts.append("## Cross-Perspective Consensus\n")
        for c in consensus:
            parts.append(f"- {c}")
        parts.append("")

    if dissents:
        parts.append("## Persistent Disagreements\n")
        for d in dissents:
            parts.append(f"- {d}")
        parts.append("")

    if platform_interps:
        parts.append("## Platform-Specific Interpretations\n")
        for platform, interp in platform_interps.items():
            parts.append(f"### {platform}\n{interp}\n")

    if echo_warnings:
        parts.append("## Bias & Limitations\n")
        for w in echo_warnings:
            parts.append(f"⚠️ {w}")
        parts.append("")

    if frameworks:
        parts.append("## Analytical Frameworks\n")
        for fw in frameworks[:5]:
            fw_type = fw.get("framework", "")
            analysis = fw.get("analysis", "")
            certainty = fw.get("certainty", "")
            parts.append(f"**{fw_type}** [{certainty}]: {analysis}\n")

    if sources:
        parts.append("## Key Sources\n")
        for s in sources[:15]:
            title = s.get("title", "(no title)")
            url = s.get("url", "")
            ts = s.get("trust_score", 0)
            parts.append(f"- [{title}]({url}) — trust: {ts:.2f}")
        parts.append("")

    return "\n".join(parts)


def synthesis_context_to_markdown(synthesis_context: Dict) -> str:
    """Simple markdown fallback when ReportEngine is unavailable."""
    return build_report_prompt(synthesis_context)
