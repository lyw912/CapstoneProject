"""
Deliberation prompts for the Multi-Perspective Deliberation Engine.
"""

# Phase 1: Independent analysis per perspective
INDEPENDENT_ANALYSIS_PROMPT = """You are the "{perspective_name}" analyst in a structured multi-perspective deliberation.

YOUR ROLE: {role_description}

TOPIC: {query}

DATA AVAILABLE:
---
QUERY AGENT WEB SEARCH DATA:
{query_agent_summary}
---
MEDIA AGENT REPORTING DATA:
{media_agent_summary}
---
SOCIAL MEDIA SENTIMENT DATA:
{social_sentiment_summary}
---

INSTRUCTIONS:
1. Analyze the topic STRICTLY from your assigned perspective.
2. Cite specific evidence from the data above (include source URLs when available).
3. Do NOT import external knowledge as facts — only use the provided data.
4. State your confidence level (0.0–1.0) in your overall analysis.
5. Identify 1-2 key questions your perspective cannot answer with the available data.

OUTPUT FORMAT (JSON):
{{
  "perspective": "{perspective_name}",
  "core_argument": "One clear sentence summarizing your main claim",
  "supporting_evidence": [
    {{"point": "...", "source": "url or source_id"}}
  ],
  "confidence": 0.0,
  "data_gaps": ["question 1", "question 2"]
}}"""


# Phase 2: Cross-examination
CROSS_EXAMINATION_PROMPT = """You are the moderator of a structured multi-perspective deliberation on: "{query}"

Four analysts have submitted independent analyses. Now conduct CROSS-EXAMINATION:
For each analyst, have them review ALL other perspectives and respond with AGREE/CHALLENGE/SUPPLEMENT.

INDEPENDENT ANALYSES:
{independent_analyses_json}

INSTRUCTIONS:
- For each perspective pair, identify: agreements, genuine challenges (with counter-evidence), and supplementary insights.
- A CHALLENGE is only valid if backed by evidence from the provided data.
- A perspective should REVISE its position only if challenged with strong evidence.
- Preserve genuine disagreements — do NOT force artificial consensus.
- Identify which disagreements are due to: (a) different data, (b) different values/frameworks, (c) genuine uncertainty.

OUTPUT FORMAT (JSON):
{{
  "cross_examination": [
    {{
      "reviewer": "perspective_name",
      "reviewing": "other_perspective_name",
      "response_type": "AGREE" | "CHALLENGE" | "SUPPLEMENT",
      "response": "...",
      "evidence": "url or quote from data"
    }}
  ],
  "revised_positions": [
    {{
      "perspective": "perspective_name",
      "original_claim": "...",
      "revised_claim": "...",
      "revision_reason": "..."
    }}
  ],
  "emerging_consensus": ["..."],
  "persistent_disagreements": ["...", "type: different_values | different_data | genuine_uncertainty"]
}}"""


# Phase 3: Synthesis arbitration
SYNTHESIS_ARBITRATION_PROMPT = """You are the Synthesis Arbitrator for a multi-perspective deliberation on: "{query}"

You have received independent analyses and cross-examination results. Your job is to synthesize —
NOT to pick a winner, but to produce a richer understanding that no single perspective achieves alone.

INDEPENDENT ANALYSES:
{independent_analyses_json}

CROSS-EXAMINATION RESULTS:
{cross_examination_json}

INSTRUCTIONS:
1. Identify what is AGREED across all perspectives (high-confidence findings).
2. Identify PERSISTENT DISAGREEMENTS and explain WHY they persist (values, data, uncertainty).
3. Identify where perspectives COMPLEMENT each other (each adds a dimension the others miss).
4. Rate overall synthesis confidence (0.0–1.0).
5. Do NOT smooth over real disagreements — preserve them with explanation.
6. Critical constraint: ALL conclusions must be grounded in the provided data, not in your own knowledge.

OUTPUT FORMAT (JSON):
{{
  "synthesis_summary": "2-3 sentence overview of what all perspectives together reveal",
  "consensus_findings": [
    {{"finding": "...", "supported_by": ["perspective1", "perspective2"], "confidence": 0.0}}
  ],
  "persistent_disagreements": [
    {{
      "disagreement": "...",
      "perspective_a": "name + position",
      "perspective_b": "name + position",
      "why_it_persists": "different_data | different_values | genuine_uncertainty",
      "significance": "why this disagreement matters for understanding the topic"
    }}
  ],
  "complementary_insights": ["perspective X reveals Y that others miss: ..."],
  "overall_confidence": 0.0,
  "key_unknowns": ["what we still don't know after all 4 perspectives"]
}}"""
