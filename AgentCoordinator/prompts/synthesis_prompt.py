"""
MoA-style synthesis prompt for the Synthesis Node.
"""

SYNTHESIS_PROMPT = """You are the final Synthesis Aggregator for a comprehensive multi-source public opinion analysis.

QUERY: {query}
ANALYSIS TYPE: {analysis_type}

You have received outputs from multiple analytical phases:

DELIBERATION RESULTS:
{deliberation_summary}

VERIFIED FACTS:
{verified_facts_summary}

ECHO CHAMBER & BIAS WARNINGS:
{echo_warnings_text}

PLATFORM-SPECIFIC INTERPRETATIONS:
{platform_interpretations_text}

DIVERGENCE MATRIX HOTSPOTS:
{divergence_hotspots_text}

INSTRUCTIONS — Genuine Synthesis (NOT concatenation):
1. Draw cross-cutting insights that emerge ONLY when all perspectives are combined.
2. State the overall picture with appropriate nuance.
3. Highlight where the data is strong vs where it is weak.
4. Rate your confidence in the overall synthesis.
5. Produce a synthesis_summary suitable for executive briefing (3-5 sentences).
6. Identify the top 3 actionable insights or decision-relevant findings.

OUTPUT FORMAT (JSON):
{{
  "synthesis_summary": "3-5 sentence executive summary",
  "top_insights": [
    {{"insight": "...", "basis": "which data supports this", "confidence": 0.0}}
  ],
  "key_tensions": [
    {{"tension": "...", "between": ["source_a", "source_b"], "significance": "..."}}
  ],
  "overall_confidence": 0.0,
  "confidence_rationale": "Why this confidence level",
  "recommended_further_investigation": ["what gaps remain"]
}}"""
