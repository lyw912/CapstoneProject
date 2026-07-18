"""
Report bridge: converts AgentCoordinator artifacts into ReportEngine inputs.

The bridge keeps the Coordinator/ReportEngine boundary explicit:
- Coordinator owns specialist fusion, evidence audit, and structured synthesis.
- ReportEngine owns template/layout/chapter rendering.

Bridge-generated labels, instructions, and report scaffolding are English.
Upstream Chinese source text should be translated when passed to ReportEngine.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from typing import Any, Dict, List, Optional


ENGLISH_OUTPUT_CONSTRAINT = (
    "Write all generated report prose, headings, captions, table labels, and "
    "explanatory text in English only. Do not output Chinese characters. "
    "Translate Chinese source material into English while preserving facts. "
    "Proper nouns and URLs may remain unchanged."
)

EVIDENCE_OUTPUT_CONSTRAINT = (
    "Treat the Binding Evidence Policy as non-negotiable. Use only Audited Findings as report "
    "assertions; never restore stronger wording removed by a proposer revision or paired judges. "
    "Rejected Claims cannot appear as facts, while contested findings, perspective tensions, and "
    "evidence gaps must remain explicitly qualified. Cite the supplied source URLs for factual claims."
)


class EvidencePolicyViolation(ValueError):
    """Raised when a generated report violates the Coordinator's binding verdicts."""


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
    this adapter projects it into two bounded evidence packages plus a compact
    trace log. The versioned Coordinator artifact remains the lossless record;
    the LLM-facing package is deliberately small enough to avoid prompt dilution.
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
        save_report=False,
        stream_handler=stream_handler,
    )
    if isinstance(result, dict):
        compliance = _validate_generated_report(coordinator_output, result.get("html_content") or "")
        result.setdefault("adapter_metadata", inputs["metadata"])
        result["adapter_metadata"]["evidence_policy"] = compliance
        if not compliance["passed"]:
            raise EvidencePolicyViolation(
                "ReportEngine output failed the binding evidence policy: "
                + "; ".join(compliance["errors"])
            )
        if save_report and hasattr(report_agent, "_save_report") and result.get("document_ir"):
            saved_files = report_agent._save_report(
                result.get("html_content") or "",
                result["document_ir"],
                result.get("report_id") or "report-evidence-bound",
            )
            result.update(saved_files)
    return result


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def _normalized_report_text(value: str) -> str:
    without_tags = _HTML_TAG_RE.sub(" ", html_lib.unescape(value or ""))
    return _SPACE_RE.sub(" ", without_tags).strip().casefold()


def _validate_generated_report(
    coordinator_output: Dict[str, Any],
    html_content: str,
) -> Dict[str, Any]:
    indexes = _evidence_indexes(coordinator_output)
    debate = indexes["debate"]
    groups = debate.get("output_groups") or {}
    material_ids = [
        str(item.get("claim_id"))
        for item in debate.get("material_claims", []) or []
        if isinstance(item, dict) and item.get("claim_id")
    ]
    reportable_ids = [
        claim_id
        for claim_id in material_ids
        if claim_id in {str(item) for item in groups.get("audited_findings", []) or []}
    ]
    if not material_ids:
        return {
            "passed": True,
            "checked_material_claims": 0,
            "checked_reportable_claims": 0,
            "errors": [],
        }

    normalized_text = _normalized_report_text(html_content)
    hrefs = {
        html_lib.unescape(match).strip()
        for match in _HREF_RE.findall(html_content or "")
    }
    errors: List[str] = []
    for claim_id in reportable_ids:
        claim = indexes["claims"].get(claim_id) or {}
        verdicts = indexes["verdicts_by_claim"].get(claim_id, [])
        wording = _claim_wording(claim, verdicts)
        if wording and _normalized_report_text(wording) not in normalized_text:
            errors.append(f"{claim_id} final wording is missing")
        citation_urls = {
            str(indexes["spans"].get(span_id, {}).get("url") or "")
            for verdict in verdicts
            for span_id in verdict.get("evidence_span_ids", []) or []
        }
        citation_urls.update(
            str(indexes["spans"].get(span_id, {}).get("url") or "")
            for span_id in claim.get("supporting_spans", []) or []
        )
        citation_urls = {
            url for url in citation_urls if url.startswith(("http://", "https://"))
        }
        if citation_urls and not any(url in hrefs for url in citation_urls):
            errors.append(f"{claim_id} has no clickable bound source")

    disabled_media = any(
        isinstance(item, dict)
        and item.get("provider") == "media_agent"
        and item.get("capability") == "specialist_llm"
        and item.get("status") == "disabled"
        for item in indexes["intelligence"].get("provider_diagnostics", []) or []
    )
    if disabled_media:
        false_attribution_patterns = [
            "media agent was responsible",
            "media agent contributed",
            "media agent retrieved",
            "media agent analyzed",
            "two specialist agents: the query agent and the media agent",
        ]
        if any(pattern in normalized_text for pattern in false_attribution_patterns):
            errors.append("disabled media_agent is credited as a contributor")

    return {
        "passed": not errors,
        "checked_material_claims": len(material_ids),
        "checked_reportable_claims": len(reportable_ids),
        "errors": errors,
    }


def build_english_report_template() -> str:
    """Return a compact template tuned for Coordinator-sourced reports."""
    return """# Multi-Source Public Opinion Analysis Report

Non-negotiable evidence policy:
- The Binding Evidence Policy in the Query package outranks narrative fluency and all pre-debate summaries.
- Use paired-judge final wording verbatim or make it narrower; never strengthen it.
- Use only Audited Findings as assertions. Label contested findings, perspective tensions, and evidence gaps; omit rejected claims from conclusions.
- Every material factual claim must include at least one clickable source URL supplied in the evidence package.
- Describe only agents/providers whose runtime status is used; do not credit disabled components.

## 1. Executive Summary
- Summarize the overall finding, confidence level, and strongest evidence.
- Use the strictest paired-judge wording for material claims.

## 2. Data and Methodology
- Explain Query/Media specialist contributions, acquisition provenance, canonicalization, stance coverage, divergence, and claim audit.
- Include limitations and data coverage.

## 3. Evidence Overview
- Present source counts, stance distribution, representative sources, and social-media voices.
- Preserve traceability to URLs and platforms.

## 4. Cross-Source Divergence and Claim Audit
- Explain divergence hotspots, accepted/weakened/rejected counts, paired adjudication, and retained counter-evidence.
- Distinguish verified facts from opinions.

## 5. Interpretation, Risks, and Recommendations
- Provide platform-aware interpretation, key tensions, bias warnings, and recommended follow-up work.

## Appendix
- Include source list, raw metric tables, and coordinator trace where useful.
"""


_VERDICT_SEVERITY = {
    "accept": 0,
    "weaken": 1,
    "unresolved": 2,
    "needs_search": 3,
    "reject": 4,
}


def _evidence_indexes(coordinator_output: Dict[str, Any]) -> Dict[str, Any]:
    intelligence = coordinator_output.get("coordinator_intelligence") or {}
    graph = intelligence.get("evidence_graph") or {}
    claims = {
        str(item.get("claim_id")): item
        for item in graph.get("claims", []) or []
        if isinstance(item, dict) and item.get("claim_id")
    }
    audits = {
        str(item.get("claim_id")): item
        for item in graph.get("audit_decisions", []) or []
        if isinstance(item, dict) and item.get("claim_id")
    }
    spans: Dict[str, Dict[str, Any]] = {}
    for evidence in graph.get("evidence_items", []) or []:
        if not isinstance(evidence, dict):
            continue
        for span in evidence.get("spans", []) or []:
            if isinstance(span, dict) and span.get("span_id"):
                spans[str(span["span_id"])] = {
                    "span": span,
                    "title": evidence.get("title") or evidence.get("source_name") or evidence.get("url") or "Source",
                    "url": evidence.get("url") or "",
                    "platform": evidence.get("platform") or "",
                    "source_type": evidence.get("source_type") or "",
                }
    debate = coordinator_output.get("debate") or intelligence.get("debate_session") or {}
    verdicts_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for verdict in debate.get("verdicts", []) or []:
        if isinstance(verdict, dict) and verdict.get("claim_id"):
            verdicts_by_claim.setdefault(str(verdict["claim_id"]), []).append(verdict)
    revisions_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for revision in debate.get("revisions", []) or []:
        if isinstance(revision, dict) and revision.get("claim_id"):
            revisions_by_claim.setdefault(str(revision["claim_id"]), []).append(revision)
    acts_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for act in debate.get("argument_acts", []) or []:
        if isinstance(act, dict) and act.get("target_claim_id"):
            acts_by_claim.setdefault(str(act["target_claim_id"]), []).append(act)
    positions_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for position in debate.get("positions", []) or []:
        if isinstance(position, dict) and position.get("claim_id"):
            positions_by_claim.setdefault(str(position["claim_id"]), []).append(position)
    return {
        "intelligence": intelligence,
        "graph": graph,
        "claims": claims,
        "audits": audits,
        "spans": spans,
        "debate": debate,
        "verdicts_by_claim": verdicts_by_claim,
        "revisions_by_claim": revisions_by_claim,
        "acts_by_claim": acts_by_claim,
        "positions_by_claim": positions_by_claim,
    }


def _strictest_verdict(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not verdicts:
        return {}
    return max(
        verdicts,
        key=lambda item: (
            _VERDICT_SEVERITY.get(str(item.get("decision") or "").lower(), 2),
            float(item.get("confidence") or 0),
        ),
    )


def _claim_wording(claim: Dict[str, Any], verdicts: List[Dict[str, Any]]) -> str:
    strictest = _strictest_verdict(verdicts)
    return str(strictest.get("final_wording") or claim.get("claim_text") or "").strip()


def _clean_markdown_text(value: Any, fallback: str = "Source") -> str:
    text = str(value or fallback).replace("\n", " ").replace("\r", " ").strip()
    return text.replace("[", "").replace("]", "")


def _claim_citations(
    claim: Dict[str, Any],
    verdicts: List[Dict[str, Any]],
    span_index: Dict[str, Dict[str, Any]],
    *,
    limit: int = 4,
) -> List[str]:
    span_ids: List[str] = []
    for verdict in verdicts:
        span_ids.extend(str(item) for item in verdict.get("evidence_span_ids", []) or [])
    span_ids.extend(str(item) for item in claim.get("supporting_spans", []) or [])
    span_ids.extend(str(item) for item in claim.get("contradicting_spans", []) or [])
    links: List[str] = []
    seen_urls = set()
    for span_id in dict.fromkeys(span_ids):
        source = span_index.get(span_id) or {}
        url = str(source.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen_urls:
            continue
        seen_urls.add(url)
        title = _clean_markdown_text(source.get("title"), source.get("platform") or span_id)
        links.append(f"[{title}]({url}) [{span_id}]")
        if len(links) >= limit:
            break
    return links


def _group_by_claim(debate: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for group, claim_ids in (debate.get("output_groups") or {}).items():
        for claim_id in claim_ids or []:
            result[str(claim_id)] = str(group)
    return result


def _build_legacy_query_engine_report(coordinator_output: Dict[str, Any]) -> str:
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
        f"- Channel-divergence pair count: {len(divergence.get('pairs', {}) or {})}",
        f"- Maximum channel divergence (TVD): {(divergence.get('max_divergence') or {}).get('pair', 'N/A')} = {(divergence.get('max_divergence') or {}).get('value', 0)}",
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


def _build_query_engine_report(coordinator_output: Dict[str, Any]) -> str:
    query = coordinator_output.get("query", "")
    source_data = coordinator_output.get("source_data", {}) or {}
    qa = source_data.get("query_agent", {}) or {}
    divergence = coordinator_output.get("divergence_matrix", {}) or {}
    bias = coordinator_output.get("bias_analysis", {}) or {}
    brief = coordinator_output.get("investigation_brief", {}) or {}
    indexes = _evidence_indexes(coordinator_output)
    claims = indexes["claims"]
    audits = indexes["audits"]
    debate = indexes["debate"]
    verdicts_by_claim = indexes["verdicts_by_claim"]
    positions_by_claim = indexes["positions_by_claim"]
    acts_by_claim = indexes["acts_by_claim"]
    revisions_by_claim = indexes["revisions_by_claim"]
    group_by_claim = _group_by_claim(debate)
    groups = debate.get("output_groups") or {}
    audited_ids = [str(item) for item in groups.get("audited_findings", []) or []]
    rejected_ids = [str(item) for item in groups.get("rejected_claims", []) or []]
    if not audited_ids:
        audited_ids = [
            claim_id
            for claim_id, item in audits.items()
            if item.get("decision") in {"accept", "weaken"}
        ]
    if not rejected_ids:
        rejected_ids = [
            claim_id
            for claim_id, item in audits.items()
            if item.get("decision") == "reject"
        ]
    material_assignments = [
        item
        for item in debate.get("material_claims", []) or []
        if isinstance(item, dict)
    ]
    material_ids = [
        str(item.get("claim_id"))
        for item in material_assignments
        if item.get("claim_id")
    ]
    prioritized_audited_ids = list(
        dict.fromkeys(
            [claim_id for claim_id in material_ids if claim_id in audited_ids]
            + audited_ids
        )
    )[:18]

    lines: List[str] = [
        f"# Evidence-Bound Query Package: {query}",
        "",
        f"Language rule: {ENGLISH_OUTPUT_CONSTRAINT}",
        "",
        "## Binding Evidence Policy",
        f"- {EVIDENCE_OUTPUT_CONSTRAINT}",
        "- The paired-judge wording shown below is the maximum-strength wording allowed. Quote it verbatim or make it narrower.",
        "- A disabled provider or agent did not contribute and must not be credited in the methodology.",
        f"- Reportable audited claim count: {len(audited_ids)}; rejected claim count: {len(rejected_ids)}.",
        "",
        "## Investigation Contract",
        f"- Original topic: {brief.get('original_query', query)}",
        f"- Factual question: {brief.get('factual_question', '')}",
        f"- Public-discourse question: {brief.get('discourse_question', '')}",
        f"- Time scope: {brief.get('time_scope', '')}",
        f"- Sample boundary: {brief.get('sample_boundary', '')}",
        "",
        "## Retrieval Metrics",
        f"- Total retained sources: {qa.get('total_sources', 0)}",
        f"- Stance coverage score: {qa.get('coverage_score', 0)}",
        f"- Channel-divergence pair count: {len(divergence.get('pairs', {}) or {})}",
        f"- Maximum channel divergence (TVD): {(divergence.get('max_divergence') or {}).get('pair', 'N/A')} = {(divergence.get('max_divergence') or {}).get('value', 0)}",
        "",
    ]

    if material_assignments:
        lines.extend(["## Material Claim Adjudication", ""])
        for assignment in material_assignments:
            claim_id = str(assignment.get("claim_id") or "")
            claim = claims.get(claim_id) or {"claim_text": claim_id}
            verdicts = verdicts_by_claim.get(claim_id, [])
            strictest = _strictest_verdict(verdicts)
            group = group_by_claim.get(claim_id, "unrouted")
            wording = _claim_wording(claim, verdicts)
            reportability = (
                "REPORTABLE ONLY AS WRITTEN"
                if group == "audited_findings"
                else "DO NOT REPORT AS A FACT"
            )
            decisions = ", ".join(
                f"{item.get('judge_id', 'judge')}={item.get('decision', 'unknown')}"
                for item in verdicts
            ) or "No paired verdict"
            opening_stances = ", ".join(
                f"{item.get('agent_id', 'agent')}={item.get('stance', 'unknown')}"
                for item in positions_by_claim.get(claim_id, [])
            ) or "No opening position"
            reviewer_reasons = list(
                dict.fromkeys(
                    str(reason)
                    for act in acts_by_claim.get(claim_id, [])
                    if act.get("actor_id") in {"skeptic", "methodologist"}
                    for reason in act.get("reason_codes", []) or []
                )
            )
            revision_types = list(
                dict.fromkeys(
                    str(item.get("revision_type") or "")
                    for item in revisions_by_claim.get(claim_id, [])
                    if item.get("revision_type")
                )
            )
            lines.extend(
                [
                    f"### {claim_id} - {reportability}",
                    f"- Output group: {group}",
                    f"- Final wording: {wording}",
                    f"- Paired decisions: {decisions}",
                    f"- Opening stances: {opening_stances}",
                    f"- Review reason codes: {', '.join(reviewer_reasons) or 'none'}",
                    f"- Proposer response: {', '.join(revision_types) or 'none'}",
                    f"- Strictest required edit: {strictest.get('required_edit', '')}",
                ]
            )
            citations = _claim_citations(claim, verdicts, indexes["spans"])
            if citations:
                lines.append(f"- Required citations: {'; '.join(citations)}")
            lines.append("")

    if prioritized_audited_ids:
        lines.extend(["## Audited Findings - Allowed Assertions", ""])
        for claim_id in prioritized_audited_ids:
            claim = claims.get(claim_id) or {"claim_text": claim_id}
            verdicts = verdicts_by_claim.get(claim_id, [])
            wording = _claim_wording(claim, verdicts)
            audit = audits.get(claim_id) or {}
            decision = (
                _strictest_verdict(verdicts).get("decision")
                or audit.get("decision")
                or "audited"
            )
            lines.append(f"- **{claim_id} [{decision}]** {wording}")
            citations = _claim_citations(claim, verdicts, indexes["spans"])
            if citations:
                lines.append(f"  - Sources: {'; '.join(citations)}")
        lines.append("")

    for group in ["contested_findings", "perspective_tensions", "evidence_gaps"]:
        claim_ids = [str(item) for item in groups.get(group, []) or []]
        if not claim_ids:
            continue
        lines.extend(
            [f"## {group.replace('_', ' ').title()} - Qualified Context Only", ""]
        )
        for claim_id in claim_ids[:8]:
            claim = claims.get(claim_id) or {}
            lines.append(
                f"- {claim_id}: "
                f"{_claim_wording(claim, verdicts_by_claim.get(claim_id, []))}"
            )
        lines.append("")

    if rejected_ids:
        rejected_material_ids = [
            claim_id for claim_id in material_ids if claim_id in rejected_ids
        ]
        lines.extend(
            [
                "## Rejected Claims - Excluded From Report Assertions",
                f"- {len(rejected_ids)} rejected claim(s) are retained only in the audit trail.",
                f"- Rejected material claim IDs: {', '.join(rejected_material_ids) or 'none'}",
                "",
            ]
        )

    stance_dist = qa.get("stance_distribution", {}) or {}
    if stance_dist:
        lines.extend(["## Stance Distribution", ""])
        for stance, value in sorted(
            stance_dist.items(), key=lambda item: -float(item[1])
        ):
            lines.append(f"- {stance}: {float(value):.1%}")
        lines.append("")

    top_sources = qa.get("top_sources", []) or []
    if top_sources:
        lines.extend(["## Top Sources", ""])
        for source in top_sources[:12]:
            title = _clean_markdown_text(source.get("title"), "(untitled)")
            url = str(source.get("url") or "")
            lines.append(
                f"- [{title}]({url}) | stance={source.get('stance', '')} "
                f"| trust={float(source.get('trust_score', 0) or 0):.2f}"
            )
        lines.append("")

    warnings = bias.get("echo_warnings", []) or []
    if warnings:
        lines.extend(["## Bias Warnings", ""])
        for warning in warnings[:12]:
            lines.append(f"- {warning}")
        lines.append("")

    provider_diagnostics = indexes["intelligence"].get("provider_diagnostics", []) or []
    if provider_diagnostics:
        lines.extend(["## Runtime Participation", ""])
        for item in provider_diagnostics:
            if not isinstance(item, dict) or item.get("capability") not in {
                "specialist_llm",
                "debate_perspective",
                "debate_evidence_review",
                "debate_adjudication",
            }:
                continue
            lines.append(
                f"- {item.get('provider', 'provider')}: "
                f"status={item.get('status', 'unknown')}, "
                f"model={item.get('model') or 'not applicable'}"
            )
        lines.append("")

    report = "\n".join(lines)
    if len(report) > 30000:
        report = (
            report[:29800].rstrip()
            + "\n\n[Bridge truncation] Additional audited records remain in the versioned Coordinator artifact."
        )
    return report


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

    dossiers = media.get("dossiers", []) or []
    if dossiers:
        lines.extend(["## Media Section Dossiers", ""])
        for dossier in dossiers:
            lines.extend(
                [
                    f"### {dossier.get('title', 'Untitled section')}",
                    str(dossier.get("summary", "")),
                    f"- Status: {dossier.get('status', 'unknown')}",
                    f"- Evidence spans: {len(dossier.get('evidence_span_ids', []) or [])}",
                    f"- Multimodal assets: {dossier.get('multimodal_asset_count', 0)}",
                    "",
                ]
            )

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
            lines.extend([
                "### Representative Social Voices",
                "Translate non-English post text into English while preserving facts and links.",
                "",
            ])
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
    debate = coordinator_output.get("debate", {}) or {}
    brief = coordinator_output.get("investigation_brief", {}) or {}
    trace = coordinator_output.get("coordinator_trace", []) or []
    indexes = _evidence_indexes(coordinator_output)
    graph = indexes["graph"]
    claim_index = {
        str(item.get("claim_id")): str(item.get("claim_text") or "")
        for item in graph.get("claims", []) or []
        if isinstance(item, dict) and item.get("claim_id")
    }
    groups = debate.get("output_groups", {}) or {}
    lines: List[str] = [
        "[Coordinator Bridge] ReportEngine handoff generated from structured Coordinator output.",
        f"[Coordinator Bridge] {ENGLISH_OUTPUT_CONSTRAINT}",
        "",
        "[Binding Evidence Policy]",
        f"- {EVIDENCE_OUTPUT_CONSTRAINT}",
        "- Paired-judge final wording is the maximum-strength wording allowed.",
        "- Do not credit disabled agents or providers as contributors.",
        "",
        "[Investigation Brief]",
        f"- Original topic: {brief.get('original_query', coordinator_output.get('query', ''))}",
        f"- Factual question: {brief.get('factual_question', '')}",
        f"- Public-discourse question: {brief.get('discourse_question', '')}",
        f"- Sample boundary: {brief.get('sample_boundary', '')}",
        "",
        "[Audited Findings]",
    ]
    audited_ids = [str(item) for item in groups.get("audited_findings", []) or []]
    for claim_id in audited_ids[:20]:
        wording = _claim_wording(
            indexes["claims"].get(claim_id)
            or {"claim_text": claim_index.get(claim_id, claim_id)},
            indexes["verdicts_by_claim"].get(claim_id, []),
        )
        lines.append(f"- {claim_id}: {wording}")
    if not audited_ids:
        for point in deliberation.get("final_consensus", []) or []:
            lines.append(f"- {point}")
    lines.append("")
    lines.append("[Contested Findings]")
    contested = [claim_index.get(str(claim_id), str(claim_id)) for claim_id in groups.get("contested_findings", []) or []]
    for point in contested or deliberation.get("final_dissents", []) or []:
        lines.append(f"- {point}")
    lines.extend(["", "[Perspective Tensions]"])
    for claim_id in groups.get("perspective_tensions", []) or []:
        lines.append(f"- {claim_index.get(str(claim_id), str(claim_id))}")
    lines.extend(["", "[Evidence Gaps]"])
    for claim_id in groups.get("evidence_gaps", []) or []:
        lines.append(f"- {claim_index.get(str(claim_id), str(claim_id))}")
    rejected = groups.get("rejected_claims", []) or []
    if rejected:
        lines.extend(["", f"[Rejected Claims] {len(rejected)} claim(s); do not use them as report assertions."])
    material_ids = [
        str(item.get("claim_id"))
        for item in debate.get("material_claims", []) or []
        if isinstance(item, dict) and item.get("claim_id")
    ]
    if material_ids:
        group_by_claim = _group_by_claim(debate)
        lines.extend(["", "[Paired Adjudication - Binding Wording]"])
        for claim_id in material_ids:
            verdicts = indexes["verdicts_by_claim"].get(claim_id, [])
            decisions = ", ".join(
                f"{item.get('judge_id', 'judge')}={item.get('decision', 'unknown')}"
                for item in verdicts
            ) or "no verdict"
            group = group_by_claim.get(claim_id, "unrouted")
            wording = _claim_wording(indexes["claims"].get(claim_id) or {}, verdicts)
            lines.append(
                f"- {claim_id} [{group}; {decisions}]: {wording}"
            )
    if trace:
        lines.extend(["", "[Coordinator Trace]"])
        lines.extend(str(entry) for entry in trace)
    return "\n".join(lines)
