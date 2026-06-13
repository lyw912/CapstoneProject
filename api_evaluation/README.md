# API Evaluation Plan for Engine Selection

This folder is independent from the main project. It is intended to benchmark candidate APIs before wiring any provider into `MediaEngine`, `QueryEngine`, `ReportEngine`, `ForumEngine`, or `MindSpider`.

## Goal

Pick the best API/provider/model for each engine based on the engine's real workload, not generic chatbot quality.

The project has two API families:

- LLM APIs: OpenAI-compatible chat completion providers such as DeepSeek, Moonshot/Kimi, Qwen/Bailian/SiliconFlow, Gemini proxy, etc.
- Search APIs: Tavily, Anspire, Bocha.

## Engine Profiles

### MediaEngine

Primary need: retrieve and synthesize broad public/media narratives.

High-value capabilities:

- Chinese web/media coverage.
- Current public opinion discovery.
- Multi-source synthesis.
- Evidence extraction with URLs.
- Robustness on noisy or conflicting search results.

Recommended benchmark mix:

- Search: Anspire vs Bocha, optionally Tavily for non-Chinese/current global topics.
- LLM: long-context synthesis, citation discipline, structured JSON output.

### QueryEngine

Primary need: stance-aware retrieval, source classification, gap filling, and structured outputs.

High-value capabilities:

- Accurate search query planning.
- Cross-source stance classification.
- JSON reliability.
- Low hallucination under source constraints.
- Timely retrieval.

Recommended benchmark mix:

- Search: Tavily as baseline, Anspire for Chinese web enhancement.
- LLM: strict JSON, source-grounded stance analysis, short reasoning latency.

### ReportEngine

Primary need: long-form report generation from upstream structured artifacts.

High-value capabilities:

- Long-context following.
- Coherent multi-section writing.
- Chart/table instruction following.
- Stable formatting in Markdown/HTML-like output.
- Low repetition and low truncation.

Recommended benchmark mix:

- LLM only.
- Emphasize long output quality, structure, and cost per finished report.

### ForumEngine

Primary need: host/moderator behavior, discussion summarization, and concise steering.

High-value capabilities:

- Fast response.
- Stable persona and moderation style.
- Concise summaries.
- Low cost for repeated calls.

Recommended benchmark mix:

- LLM only.
- Favor latency/cost more than maximum reasoning depth.

### MindSpider

Primary need: hot-topic extraction and summary from news lists.

High-value capabilities:

- Keyword extraction.
- Clustering/summarization.
- Good Chinese handling.
- Cheap batch processing.

Recommended benchmark mix:

- LLM only, optionally with search if the upstream news fetcher is replaced.
- Favor structured keyword output and cost.

## Scoring Model

Use a 0-5 score for each metric:

- 5: excellent, production-ready.
- 4: good, minor issues.
- 3: usable but needs guardrails.
- 2: weak for this engine.
- 1: mostly unusable.
- 0: failed call, invalid output, or unsafe result.

### Common LLM Metrics

- `task_success`: Did it answer the task correctly?
- `format_compliance`: Did it follow required JSON/Markdown/schema?
- `grounding`: Did it avoid unsupported claims?
- `reasoning_quality`: Are classifications and synthesis defensible?
- `language_quality`: Is Chinese/English output fluent for the requested language?
- `latency`: Relative score from measured seconds.
- `cost`: Relative score from estimated token cost.
- `stability`: Consistency across retries.

### Common Search Metrics

- `relevance`: Are top results relevant?
- `freshness`: Are current-event queries fresh?
- `source_quality`: Are sources authoritative/diverse?
- `coverage`: Does it return enough useful results?
- `metadata_quality`: Dates/snippets/images/structured cards available?
- `latency`: Relative score from measured seconds.
- `cost`: Relative score from provider pricing.
- `parseability`: Is the response easy to normalize?

## Suggested Engine Weights

Weights are in `engine_profiles.json`. Adjust them based on your actual priorities.

Typical defaults:

- MediaEngine LLM: grounding and synthesis quality dominate.
- MediaEngine Search: relevance, Chinese coverage, freshness dominate.
- QueryEngine LLM: JSON compliance and classification accuracy dominate.
- QueryEngine Search: relevance, freshness, and metadata dominate.
- ReportEngine LLM: long-form structure and language quality dominate.
- ForumEngine LLM: latency/cost and concise quality dominate.
- MindSpider LLM: keyword extraction and Chinese handling dominate.

## Workflow

1. Copy `providers.example.json` to `providers.local.json`.
2. Fill candidate APIs, base URLs, models, and keys.
3. Run the benchmark:

```powershell
cd D:\huang\Desktop\Project\api_evaluation
python run_evaluation.py --providers providers.local.json --out results
```

4. Review:

- `results/raw_results.jsonl`: every raw API call result.
- `results/summary.csv`: provider scores per engine/task.
- `results/recommendations.md`: ranked recommendation per engine.

## Important Evaluation Rules

- Test each provider with the same prompts and same temperature.
- Run at least 3 repetitions for finalists.
- Record failures, invalid JSON, and timeouts as real signal.
- Use representative Chinese public-opinion topics, not generic trivia.
- Separate search API quality from LLM quality.
- Do not pick a provider based on one spectacular answer. Prefer stable performance.

## Interpreting Results

For each engine, choose:

- Best quality provider if quality gap is large.
- Cheapest acceptable provider if quality scores are close.
- Fastest acceptable provider for ForumEngine-style repeated calls.
- Provider with best JSON/schema reliability for QueryEngine.
- Provider with best long-context output for ReportEngine.

Final selection should look like:

| Engine | LLM API | Search API | Reason |
| --- | --- | --- | --- |
| MediaEngine | model/provider A | Anspire/Bocha | best Chinese retrieval + grounded synthesis |
| QueryEngine | model/provider B | Tavily + Anspire | best JSON + stance classification |
| ReportEngine | model/provider C | N/A | best long report quality |
| ForumEngine | model/provider D | N/A | fast and cheap |
| MindSpider | model/provider E | N/A | good keyword extraction at low cost |
