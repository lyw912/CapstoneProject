"""
AcademicReportGenerator: transform coordinator_output.json into an
English, evidence-traced Markdown report.

The generator is intentionally deterministic and does not call an LLM. It is
used as the Coordinator's local fallback report and as a readable handoff
artifact for ReportEngine.
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_academic_report(coordinator_output: Dict[str, Any]) -> str:
    """
    Build an English academic-style Markdown report from Coordinator output.

    Raw Chinese social media text is preserved only when it is used as original
    evidence. All section headings, explanations, metrics, and generated prose
    are English.
    """
    lines: List[str] = []
    write = lines.append

    query = coordinator_output.get("query", "")
    analysis_type = coordinator_output.get("analysis_type", "general")
    generated_at = coordinator_output.get("generated_at", "")
    duration = float(coordinator_output.get("pipeline_duration_seconds", 0) or 0)

    divergence = coordinator_output.get("divergence_matrix", {}) or {}
    deliberation = coordinator_output.get("deliberation", {}) or {}
    platform_interps = coordinator_output.get("platform_interpretations", {}) or {}
    bias = coordinator_output.get("bias_analysis", {}) or {}
    fact_opinion = coordinator_output.get("fact_opinion_separation", {}) or {}
    synthesis = coordinator_output.get("synthesis", {}) or {}
    source_data = coordinator_output.get("source_data", {}) or {}
    qa_data = source_data.get("query_agent", {}) or {}
    media_data = source_data.get("media_agent", {}) or {}
    social = qa_data.get("social_sentiment") or {}
    trace = coordinator_output.get("coordinator_trace", []) or []

    confidence = float(synthesis.get("overall_confidence", 0.5) or 0.5)
    total_sources = int(qa_data.get("total_sources", 0) or 0)
    stance_dist = qa_data.get("stance_distribution", {}) or {}
    top_sources = qa_data.get("top_sources", []) or []
    platforms_queried = social.get("platforms_queried", []) if social else []
    total_posts = int(social.get("total_posts", 0) or 0) if social else 0
    total_comments = int(social.get("total_comments", 0) or 0) if social else 0
    cssd_score = float(social.get("divergence_score", 0) or 0) if social else 0.0

    consensus_count = len(deliberation.get("final_consensus", []) or [])
    dissent_count = len(deliberation.get("final_dissents", []) or [])
    fact_count = len(fact_opinion.get("verified_facts", []) or [])
    hotspot_count = len(divergence.get("hotspots", []) or [])
    pairs_count = len(divergence.get("pairs", {}) or {})

    write("# Multi-Source Public Opinion Analysis Report")
    write("")
    write(f"**Research Query:** {query}")
    write(f"**Analysis Type:** {analysis_type}")
    write(f"**Generated At:** {generated_at}")
    write(f"**Pipeline Runtime:** {duration:.1f}s")
    write(f"**Overall Confidence:** {confidence:.0%}")
    write("")
    write("---")
    write("")

    write("## Abstract")
    write("")
    summary_text = synthesis.get("summary", "")
    if summary_text:
        write(str(summary_text))
        write("")

    data_sentence = (
        f"This report synthesizes {total_sources} web search sources"
        f", {total_posts} social media posts, and {total_comments} comments"
        if total_posts or total_comments
        else f"This report synthesizes {total_sources} web search sources"
    )
    write(
        f"{data_sentence}. The Coordinator ran a structured deliberation across "
        f"{len(deliberation.get('perspectives_used', []) or [])} analytical perspectives, "
        f"identified {consensus_count} cross-perspective consensus points and "
        f"{dissent_count} persistent disagreements, extracted {fact_count} verifiable facts, "
        f"and calculated {pairs_count} source-pair divergence scores with "
        f"{hotspot_count} notable hotspots."
    )
    write("")
    write("---")
    write("")

    write("## 1. Introduction and Background")
    write("")
    write(
        f"This report analyzes the public-opinion landscape around **{query}** through "
        "multi-source retrieval, structured deliberation, source divergence measurement, "
        "fact-opinion separation, and evidence-traced reporting."
    )
    write("")
    write("The analysis pipeline contributes five capabilities:")
    write("")
    write("1. **Cross-source triangulation:** compares web/media evidence with available social discussion.")
    write("2. **Structured multi-perspective deliberation:** separates analytical viewpoints before synthesis.")
    write("3. **Cross-Source Sentiment Difference (CSSD):** quantifies stance divergence among sources.")
    write("4. **Fact-opinion separation:** distinguishes verifiable claims from interpretations and sentiment.")
    write("5. **Bias and echo-chamber review:** flags unusually narrow or coordinated evidence patterns.")
    write("")
    write("---")
    write("")

    write("## 2. Methodology and Metrics")
    write("")
    write("### 2.1 Data Acquisition Layers")
    write("")
    write("| Layer | Sources | Role |")
    write("|---|---|---|")
    write("| Web and news retrieval | Tavily and Anspire | Broad, traceable coverage of media and institutional narratives |")
    write("| Social-media retrieval | MindSpider database when available | Native platform posts, comments, and public reactions |")
    write("| Media analysis | MediaAgent | Chinese media framing and multimodal context |")
    write("")
    write("### 2.2 Core Metrics")
    write("")
    write("**CSSD (Cross-Source Sentiment Difference).** CSSD measures the distance between two normalized stance distributions. A higher value indicates stronger divergence.")
    write("")
    write("`CSSD(A, B) = 1 - cosine(stance_vector_A, stance_vector_B)`")
    write("")
    write("| CSSD Range | Interpretation |")
    write("|---|---|")
    write("| 0.0-0.1 | Nearly identical |")
    write("| 0.1-0.3 | Low divergence |")
    write("| 0.3-0.6 | Moderate divergence |")
    write("| 0.6-0.8 | High divergence |")
    write("| 0.8-1.0 | Extreme divergence |")
    write("")
    coverage = float(qa_data.get("coverage_score", 0) or 0)
    write("**SCS (Stance Coverage Score).** SCS estimates whether retrieved evidence covers the major stance categories. Higher values indicate broader stance coverage.")
    write("")
    write(f"Current SCS: **{coverage:.2f}** ({_scs_label(coverage)}).")
    write("")
    write("**TrustScore.** TrustScore combines domain authority, timeliness, content quality, and retrieval relevance into a 0-1 source reliability score.")
    write("")
    write("### 2.3 Deliberation Protocol")
    write("")
    perspectives_used = deliberation.get("perspectives_used", []) or []
    write(
        f"For this `{analysis_type}` query, the Coordinator selected "
        f"{len(perspectives_used)} analytical perspectives:"
    )
    write("")
    for perspective in perspectives_used:
        write(f"- {perspective}")
    if not perspectives_used:
        write("- General evidence review")
    write("")
    write("The deliberation proceeds through independent analysis, cross-examination, and synthesis arbitration. The design preserves meaningful disagreement instead of forcing artificial consensus.")
    write("")
    write("---")
    write("")

    write("## 3. Data Overview")
    write("")
    write("### 3.1 Web Search Sources")
    write("")
    write(f"The QueryAgent retained **{total_sources}** deduplicated sources. The stance coverage score is **{coverage:.2f}**.")
    write("")
    if stance_dist:
        write("| Stance | Share | Meaning |")
        write("|---|---:|---|")
        for stance, ratio in sorted(stance_dist.items(), key=lambda item: -float(item[1])):
            write(f"| {stance} | {float(ratio):.1%} | {_stance_meaning(stance)} |")
        write("")

    if social and social.get("mode") == "available":
        write("### 3.2 Social Media Data")
        write("")
        write(
            f"The social layer covers **{', '.join(platforms_queried)}**, with "
            f"**{total_posts} posts** and **{total_comments} comments**."
        )
        write("")
        social_dist = social.get("sentiment_distribution", {}) or {}
        if social_dist:
            write("| Stance | Share |")
            write("|---|---:|")
            for stance, ratio in sorted(social_dist.items(), key=lambda item: -float(item[1])):
                write(f"| {stance} | {float(ratio):.1%} |")
            write("")
        write(f"Web/social CSSD: **{cssd_score:.3f}** ({_cssd_label(cssd_score)}).")
        write("")

        per_platform = social.get("per_platform", {}) or {}
        if per_platform:
            write("| Platform | Posts | Dominant Stance | Distribution |")
            write("|---|---:|---|---|")
            for platform, stats in per_platform.items():
                if not stats:
                    continue
                count = stats.get("count", 0) or stats.get("post_count", 0)
                dist = stats.get("distribution", {}) or {}
                dominant = max(dist, key=dist.get) if dist else "-"
                dominant_pct = float(dist.get(dominant, 0) or 0) if dist else 0.0
                dist_str = ", ".join(
                    f"{name}: {float(value):.0%}"
                    for name, value in sorted(dist.items(), key=lambda item: -float(item[1]))
                    if float(value) > 0.01
                )
                write(f"| {platform} | {count} | {dominant} ({dominant_pct:.0%}) | {dist_str} |")
            write("")

    write("---")
    write("")

    write("## 4. Findings")
    write("")
    write("This section separates traceable evidence from interpretation so that readers can evaluate the basis of each conclusion.")
    write("")

    verified_facts = fact_opinion.get("verified_facts", []) or []
    if verified_facts:
        write("### 4.1 Verified Facts")
        write("")
        for idx, fact in enumerate(verified_facts, 1):
            status = fact.get("verification_status", "unknown")
            conf = float(fact.get("confidence", 0) or 0)
            write(f"**Fact {idx}** [{_verification_label(status)}, confidence {conf:.0%}]")
            write("")
            write(f"> {fact.get('fact', '')}")
            write("")
            sources_list = fact.get("sources", []) or []
            if sources_list:
                write(f"*Sources:* {'; '.join(str(source) for source in sources_list)}")
                write("")

    write("### 4.2 Cross-Source Divergence Matrix")
    write("")
    max_div = divergence.get("max_divergence", {}) or {}
    min_div = divergence.get("min_divergence", {}) or {}
    write(
        f"The Coordinator calculated CSSD for **{len(divergence.get('pairs', {}) or {})}** "
        f"source pairs. The highest divergence is **{max_div.get('pair', '')}** "
        f"at **{float(max_div.get('value', 0) or 0):.3f}**; the lowest is "
        f"**{min_div.get('pair', '')}** at **{float(min_div.get('value', 0) or 0):.3f}**."
    )
    write("")
    write("`[VISUALIZATION PLACEHOLDER: Cross-source CSSD heatmap]`")
    write("")

    hotspots = divergence.get("hotspots", []) or []
    if hotspots:
        write("Notable divergence hotspots:")
        write("")
        for hotspot in hotspots:
            write(f"- {hotspot}")
        write("")

    write("### 4.3 Representative Sources")
    write("")
    if top_sources:
        by_stance: Dict[str, List[Dict[str, Any]]] = {}
        for source in top_sources:
            by_stance.setdefault(source.get("stance", "unclassified"), []).append(source)
        for stance in ["official", "support", "oppose", "neutral", "background", "unclassified"]:
            items = by_stance.get(stance, [])
            if not items:
                continue
            write(f"**{stance.upper()} sources** ({len(items)} shown):")
            write("")
            for source in items[:3]:
                title = source.get("title", "(untitled)")
                url = source.get("url", "")
                trust_score = float(source.get("trust_score", 0) or 0)
                write(f"- [{title}]({url}) - TrustScore: {trust_score:.2f}")
            write("")

    voices = social.get("top_social_voices", []) if social else []
    if voices:
        write("### 4.4 Representative Social Media Voices")
        write("")
        write("Original-language social posts are preserved as primary evidence.")
        write("")
        for voice in voices[:6]:
            platform = voice.get("platform", "")
            stance = voice.get("stance", "")
            content = (voice.get("content", "") or "")[:240]
            url = voice.get("url", "")
            publish_time = voice.get("publish_time", "")
            write(f"**[{platform}]** [{stance}] - {publish_time}")
            write("")
            write(f"> {content}")
            write("")
            if url:
                write(f"[Original post]({url})")
                write("")

    comment_data = social.get("comment_sentiment", {}) if social else {}
    top_comments = comment_data.get("top_comments", []) if comment_data else []
    if top_comments:
        write("### 4.5 Comment-Level Sentiment")
        write("")
        comment_total = int(comment_data.get("total", 0) or 0)
        write(f"The comment layer includes **{comment_total}** analyzed comments.")
        write("")
        comment_dist = comment_data.get("distribution", {}) or {}
        if comment_dist:
            write("| Stance | Share |")
            write("|---|---:|")
            for stance, ratio in sorted(comment_dist.items(), key=lambda item: -float(item[1])):
                write(f"| {stance} | {float(ratio):.0%} |")
            write("")
        for comment in top_comments[:5]:
            platform = comment.get("platform", "")
            stance = comment.get("stance", "")
            likes = comment.get("like_count", 0)
            content = (comment.get("content", "") or "")[:220]
            write(f"- **[{platform}] [{stance}] likes={likes}**")
            write(f"  > {content}")
        write("")

    write("---")
    write("")

    write("## 5. Multi-Perspective Deliberation")
    write("")
    write("`[INTERACTIVE PLACEHOLDER: Collapsible deliberation timeline]`")
    write("")
    phases = deliberation.get("phases", []) or []
    for phase_data in phases:
        phase_name = phase_data.get("phase", "")
        title = {
            "independent": "Independent Analysis",
            "cross_examination": "Cross-Examination",
            "synthesis_arbitration": "Synthesis Arbitration",
        }.get(phase_name, phase_name.replace("_", " ").title())
        write(f"### {title}")
        write("")
        summary = phase_data.get("summary", "")
        if summary:
            write(str(summary))
            write("")
        for label, key in (("Consensus", "consensus_points"), ("Dissent", "dissent_points")):
            points = phase_data.get(key, []) or []
            if points:
                write(f"**{label}:**")
                write("")
                for point in points[:5]:
                    write(f"- {point}")
                write("")

    final_consensus = _deduplicate_strings(deliberation.get("final_consensus", []) or [])
    final_dissents = _deduplicate_strings(deliberation.get("final_dissents", []) or [])
    if final_consensus:
        write("### Cross-Perspective Consensus")
        write("")
        for idx, point in enumerate(final_consensus, 1):
            write(f"{idx}. {point}")
        write("")
    if final_dissents:
        write("### Persistent Disagreements")
        write("")
        for idx, point in enumerate(final_dissents, 1):
            write(f"{idx}. {point}")
        write("")
    write(f"Deliberation confidence: **{float(deliberation.get('confidence', 0) or 0):.0%}**")
    write("")
    write("---")
    write("")

    write("## 6. Bias Assessment and Information Integrity")
    write("")
    echo_warnings = bias.get("echo_warnings", []) or []
    silent = bias.get("silent_majority_hypothesis")
    if echo_warnings:
        write("### Echo-Chamber and Bias Warnings")
        write("")
        for warning in echo_warnings:
            write(f"- {warning}")
        write("")
    if silent:
        write("### Silent-Majority Hypothesis")
        write("")
        write(f"> {silent}")
        write("")
    if not echo_warnings and not silent:
        write("No material bias warning was strong enough to affect the main conclusions.")
        write("")
    write("---")
    write("")

    write("## 7. Conclusions and Implications")
    write("")
    if platform_interps:
        write("### Platform-Aware Interpretation")
        write("")
        for platform, interp in platform_interps.items():
            if not interp:
                continue
            write(f"#### {platform.title()}")
            write("")
            write(str(interp))
            write("")

    frameworks = fact_opinion.get("analytical_frameworks", []) or []
    if frameworks:
        write("### Analytical Frameworks")
        write("")
        for framework in frameworks:
            framework_name = framework.get("framework", "")
            certainty = framework.get("certainty", "")
            analysis = framework.get("analysis", "")
            write(f"**{framework_name.title()} perspective** [certainty: {certainty}]")
            write("")
            write(f"> {analysis}")
            write("")

    tensions = synthesis.get("key_tensions", []) or []
    if tensions:
        write("### Key Tensions")
        write("")
        for tension in tensions:
            between = " vs ".join(str(item) for item in tension.get("between", []) or [])
            write(f"**{between}**")
            write("")
            write(f"- Tension: {tension.get('tension', '')}")
            write(f"- Significance: {tension.get('significance', '')}")
            write("")

    write("### Limitations")
    write("")
    limitations = [
        f"The retained web evidence contains {total_sources} sources, so rare or low-visibility views may remain underrepresented.",
        "LLM-based classification and deliberation may introduce model bias despite structured prompts and evidence constraints.",
    ]
    if total_posts or total_comments:
        limitations.append(
            f"The social layer contains {total_posts} posts and {total_comments} comments from the queried platform set; it should not be treated as a complete census of public opinion."
        )
    if media_data.get("mode") == "test_data":
        limitations.append("MediaAgent content came from pipeline validation data, so media-specific conclusions should be treated as provisional.")
    for idx, limitation in enumerate(limitations, 1):
        write(f"{idx}. {limitation}")
    write("")

    recommended = synthesis.get("recommended_investigation", []) or []
    if recommended:
        write("### Recommended Follow-Up")
        write("")
        for item in recommended:
            write(f"- {item}")
        write("")

    write("`[VISUALIZATION PLACEHOLDER: Chapter-level confidence dashboard]`")
    write("")
    write("---")
    write("")

    write("## Appendices")
    write("")
    if top_sources:
        write("### Appendix A: Source List")
        write("")
        write("<details>")
        write(f"<summary>Show {len(top_sources)} top sources</summary>")
        write("")
        write("| # | Stance | TrustScore | Title | Link |")
        write("|---:|---|---:|---|---|")
        for idx, source in enumerate(top_sources, 1):
            title = (source.get("title", "") or "")[:80]
            url = source.get("url", "")
            trust_score = float(source.get("trust_score", 0) or 0)
            stance = source.get("stance", "")
            write(f"| {idx} | {stance} | {trust_score:.2f} | {title} | [link]({url}) |")
        write("")
        write("</details>")
        write("")

    pairs_data = divergence.get("pairs", {}) or {}
    if pairs_data:
        write("### Appendix B: CSSD Matrix Raw Data")
        write("")
        write("<details>")
        write("<summary>Show all CSSD values</summary>")
        write("")
        write("| Source Pair | CSSD | Level |")
        write("|---|---:|---|")
        for pair, value in sorted(pairs_data.items(), key=lambda item: -float(item[1])):
            write(f"| {str(pair).replace('|', ' <-> ')} | {float(value):.4f} | {_cssd_label(float(value))} |")
        write("")
        write("</details>")
        write("")

    if trace:
        write("### Appendix C: Coordinator Trace")
        write("")
        write("<details>")
        write("<summary>Show execution trace</summary>")
        write("")
        write("```text")
        for entry in trace:
            write(str(entry))
        write("```")
        write("")
        write("</details>")
        write("")

    write("### Appendix D: Planned ReportEngine Enhancements")
    write("")
    write("- Render the CSSD matrix as an interactive heatmap.")
    write("- Render deliberation phases as collapsible timeline sections.")
    write("- Surface confidence annotations at chapter level.")
    write("")
    write("---")
    write("")
    write("*Generated by AgentCoordinator.*")
    write(f"*Schema version: {coordinator_output.get('schema_version', '1.0')}.*")

    return "\n".join(lines)


def _deduplicate_strings(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = str(item)[:80].strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(str(item))
    return result


def _scs_label(scs: float) -> str:
    if scs >= 0.95:
        return "complete coverage"
    if scs >= 0.7:
        return "high coverage"
    if scs >= 0.5:
        return "moderate coverage"
    return "low coverage"


def _cssd_label(cssd: float) -> str:
    if cssd < 0.1:
        return "nearly identical"
    if cssd < 0.3:
        return "low divergence"
    if cssd < 0.6:
        return "moderate divergence"
    if cssd < 0.8:
        return "high divergence"
    return "extreme divergence"


def _verification_label(status: str) -> str:
    return {
        "cross_verified": "cross-verified",
        "single_source": "single-source",
        "disputed": "disputed",
    }.get(status, status or "unknown")


def _stance_meaning(stance: str) -> str:
    return {
        "official": "Institutional or authoritative statement",
        "support": "Supportive or positive assessment",
        "oppose": "Critical or skeptical assessment",
        "neutral": "Neutral or analytical discussion",
        "background": "Background context or technical explanation",
    }.get(stance, stance)
