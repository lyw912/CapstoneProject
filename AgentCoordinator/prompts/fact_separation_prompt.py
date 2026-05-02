"""
Fact-Opinion Separation prompt for Echo Chamber Breaker.
"""

FACT_OPINION_SEPARATION_PROMPT = """You are a critical analyst performing Fact-Opinion Separation.

QUERY: {query}

SYNTHESIS FROM DELIBERATION:
{synthesis_summary}

ALL SOURCE DATA SUMMARY:
{all_data_summary}

INSTRUCTIONS:
Strictly separate the following three categories. Each item must be from the provided data, not invented.

OUTPUT FORMAT (JSON):
{{
  "verified_facts": [
    {{
      "fact": "Precise factual statement (numbers, dates, official statements, technical specs)",
      "sources": ["url1", "url2"],
      "verification_status": "cross_verified" | "single_source" | "disputed",
      "confidence": 0.0
    }}
  ],
  "opinions_and_sentiments": [
    {{
      "perspective": "Summary of this opinion/stance",
      "holders": "Description of who holds this view (platform, demographic, group)",
      "sentiment_intensity": "strong" | "moderate" | "mild",
      "platform_distribution": {{"weibo": 0.0, "zhihu": 0.0}},
      "potential_biases": ["echo_chamber", "astroturfing", "algorithm_filter", "elite_bias"]
    }}
  ],
  "analytical_frameworks": [
    {{
      "framework": "economic" | "technical" | "historical" | "sociological" | "political",
      "analysis": "The analytical insight derived from applying this framework",
      "basis": "Which verified facts support this analysis",
      "certainty": "high" | "medium" | "speculative"
    }}
  ]
}}"""
