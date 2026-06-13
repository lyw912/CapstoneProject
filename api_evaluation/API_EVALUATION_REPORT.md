# API Evaluation Report

Date: 2026-06-14  
Scope: Independent API evaluation for MediaEngine, QueryEngine, ReportEngine, ForumEngine, and MindSpider.  
Evaluation folder: `D:\huang\Desktop\Project\api_evaluation`

## Executive Summary

This evaluation tested three OpenAI-compatible LLM candidates and one Chinese media/web search API:

- `deepseek-chat`
- `deepseek-reasoner`
- `qwen-plus-compatible`
- `bocha` search

Two evaluation rounds were completed:

- Smoke test: 1 repetition per test case.
- Full repeat test: 3 repetitions per test case.

The full repeat test completed successfully with no failed calls. All LLM providers and Bocha returned `success_rate=1.0`.

Recommended quality-oriented assignment:

| Engine | Recommended API | Reason |
| --- | --- | --- |
| MediaEngine LLM | `qwen-plus-compatible` | Highest score for Chinese media synthesis, but only slightly ahead of `deepseek-chat`. |
| MediaEngine Search | `bocha` | Stable Chinese media/web retrieval quality. |
| QueryEngine LLM | `deepseek-chat` | Best score, fastest among strong candidates, good JSON/task compliance. |
| QueryEngine Search | `bocha` | Stable relevance and source coverage for Chinese retrieval tasks. |
| ReportEngine LLM | `deepseek-chat` | Best score and much faster than Qwen on report tasks. |
| ForumEngine LLM | `qwen-plus-compatible` or `deepseek-chat` | Qwen scored slightly higher; DeepSeek Chat is faster and nearly tied. |
| MindSpider LLM | `qwen-plus-compatible` | Best score for keyword/topic extraction and clustering tasks. |
| MindSpider Search | `bocha` | Stable retrieval, but currently only one MindSpider search case was included. |

Recommended pragmatic deployment assignment:

| Engine | API |
| --- | --- |
| MediaEngine | `qwen-plus-compatible` |
| QueryEngine | `deepseek-chat` |
| ReportEngine | `deepseek-chat` |
| ForumEngine | `deepseek-chat` |
| MindSpider | `qwen-plus-compatible` |
| Search | `bocha` |

If simplicity, latency, and cost matter more than small quality differences, use `deepseek-chat` for all LLM engines and `bocha` for search.

## Evaluation Method

The evaluation used the standalone harness:

- `run_evaluation.py`
- `providers.local.json`
- `test_cases.json`
- `engine_profiles.json`

The harness performs:

- OpenAI-compatible chat completion calls for LLM providers.
- Bocha search API calls for Chinese media/web retrieval.
- Automatic scoring using engine-specific weights.
- JSON/format checks for structured-output tasks.
- Required-term coverage checks.
- Forbidden-pattern checks for hallucination-prone tasks.
- Latency measurement.
- CSV and Markdown report generation.

The scoring scale is 0-5. Weighted engine-level scores are computed from task metrics such as:

- task success
- format compliance
- grounding
- reasoning quality
- language quality
- latency
- cost placeholder
- stability placeholder

Search scoring includes:

- relevance
- freshness
- source quality
- coverage
- metadata quality
- latency
- parseability

Important limitation: this report is based on automatic scoring plus light inspection. Final production selection should still include manual review of representative outputs, especially `ReportEngine` long-form outputs and `QueryEngine` JSON classification outputs.

## Test Coverage

The LLM test set covers:

- MediaEngine: Chinese public-opinion synthesis, rumor/noise filtering, evidence boundary control.
- QueryEngine: stance classification, retrieval planning, gap filling, no-hallucination behavior.
- ReportEngine: long-form Chinese report generation, chart/table planning, conservative revision.
- ForumEngine: multi-agent discussion moderation and concise intervention.
- MindSpider: keyword extraction, topic clustering, duplicate topic merge.

The search test set covers:

- Chinese public-opinion retrieval.
- Low-altitude economy policy retrieval.
- Food safety rumor/regulatory response retrieval.
- Brand recall and consumer-rights retrieval.
- Social platform complaint retrieval.
- Macro/structured signal retrieval.
- English global context retrieval.
- Disaster response retrieval.

## Full Repeat Test Results

Source: `results_full_r3\summary.csv`

| Engine | Type | Provider | Score | Success Rate | Avg Latency | Cases |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ForumEngine | LLM | `deepseek-chat` | 4.238 | 1.0 | 1.896s | 6 |
| ForumEngine | LLM | `deepseek-reasoner` | 4.224 | 1.0 | 3.175s | 6 |
| ForumEngine | LLM | `qwen-plus-compatible` | 4.269 | 1.0 | 3.235s | 6 |
| MediaEngine | LLM | `deepseek-chat` | 4.407 | 1.0 | 3.196s | 6 |
| MediaEngine | LLM | `deepseek-reasoner` | 4.351 | 1.0 | 5.068s | 6 |
| MediaEngine | LLM | `qwen-plus-compatible` | 4.418 | 1.0 | 5.673s | 6 |
| MediaEngine | Search | `bocha` | 4.646 | 1.0 | 14.399s | 18 |
| MindSpider | LLM | `deepseek-chat` | 4.255 | 1.0 | 1.993s | 6 |
| MindSpider | LLM | `deepseek-reasoner` | 4.243 | 1.0 | 3.991s | 6 |
| MindSpider | LLM | `qwen-plus-compatible` | 4.378 | 1.0 | 5.217s | 6 |
| MindSpider | Search | `bocha` | 4.709 | 1.0 | 13.840s | 3 |
| QueryEngine | LLM | `deepseek-chat` | 4.505 | 1.0 | 2.423s | 9 |
| QueryEngine | LLM | `deepseek-reasoner` | 4.399 | 1.0 | 5.612s | 9 |
| QueryEngine | LLM | `qwen-plus-compatible` | 4.466 | 1.0 | 6.286s | 9 |
| QueryEngine | Search | `bocha` | 4.668 | 1.0 | 13.006s | 18 |
| ReportEngine | LLM | `deepseek-chat` | 4.227 | 1.0 | 5.479s | 9 |
| ReportEngine | LLM | `deepseek-reasoner` | 4.196 | 1.0 | 9.155s | 9 |
| ReportEngine | LLM | `qwen-plus-compatible` | 4.167 | 1.0 | 12.373s | 9 |

## Score Stability

Source: grouped statistics from `results_full_r3\manual_review.csv`.

| Group | Min | Max | Avg | Count |
| --- | ---: | ---: | ---: | ---: |
| MediaEngine / LLM / `deepseek-chat` | 4.000 | 4.669 | 4.407 | 6 |
| QueryEngine / LLM / `deepseek-chat` | 4.104 | 4.705 | 4.505 | 9 |
| ReportEngine / LLM / `deepseek-chat` | 3.893 | 4.634 | 4.227 | 9 |
| ForumEngine / LLM / `deepseek-chat` | 3.734 | 4.570 | 4.238 | 6 |
| MindSpider / LLM / `deepseek-chat` | 3.805 | 4.705 | 4.255 | 6 |
| MediaEngine / LLM / `deepseek-reasoner` | 3.978 | 4.656 | 4.351 | 6 |
| QueryEngine / LLM / `deepseek-reasoner` | 4.041 | 4.688 | 4.399 | 9 |
| ReportEngine / LLM / `deepseek-reasoner` | 3.827 | 4.614 | 4.196 | 9 |
| ForumEngine / LLM / `deepseek-reasoner` | 3.755 | 4.545 | 4.224 | 6 |
| MindSpider / LLM / `deepseek-reasoner` | 3.781 | 4.705 | 4.243 | 6 |
| MediaEngine / LLM / `qwen-plus-compatible` | 4.206 | 4.643 | 4.418 | 6 |
| QueryEngine / LLM / `qwen-plus-compatible` | 4.087 | 4.666 | 4.466 | 9 |
| ReportEngine / LLM / `qwen-plus-compatible` | 3.832 | 4.553 | 4.167 | 9 |
| ForumEngine / LLM / `qwen-plus-compatible` | 3.806 | 4.560 | 4.269 | 6 |
| MindSpider / LLM / `qwen-plus-compatible` | 4.067 | 4.694 | 4.378 | 6 |
| MediaEngine / Search / `bocha` | 4.444 | 4.778 | 4.646 | 18 |
| QueryEngine / Search / `bocha` | 4.466 | 4.882 | 4.668 | 18 |
| MindSpider / Search / `bocha` | 4.632 | 4.778 | 4.709 | 3 |

## Smoke Test Comparison

The smoke test and full repeat test produced broadly consistent conclusions:

- `deepseek-chat` remained strongest for QueryEngine and ReportEngine.
- `qwen-plus-compatible` remained strongest for MediaEngine and MindSpider.
- `bocha` remained viable for search after credential correction.
- `deepseek-reasoner` did not show enough quality gain to justify its higher latency in this test set.

The only material change was ForumEngine:

- Smoke test ranked `deepseek-reasoner` first.
- Full repeat test ranked `qwen-plus-compatible` first.
- The score gap is small, and `deepseek-chat` is fastest.

For ForumEngine, the practical recommendation is therefore `deepseek-chat` unless the team prioritizes slightly richer moderation language over latency.

## Provider Observations

### DeepSeek Chat

Strengths:

- Best overall pragmatic choice.
- Fastest LLM candidate in most engine groups.
- Highest score for QueryEngine and ReportEngine.
- Strong enough for all LLM engines.

Weaknesses:

- Slightly below Qwen Plus for MediaEngine and MindSpider in automatic scoring.

Best use:

- QueryEngine
- ReportEngine
- ForumEngine if latency/cost matter
- Whole-system default if operational simplicity matters

### DeepSeek Reasoner

Strengths:

- Stable success rate.
- Reasoning-oriented model may still be useful for special cases outside this test set.

Weaknesses:

- Did not win any engine group in the full repeat test.
- Higher latency than `deepseek-chat`.

Best use:

- Not recommended as default.
- Keep as optional fallback for difficult reasoning tasks if later manual tests show clear advantage.

### Qwen Plus

Strengths:

- Best score for MediaEngine, ForumEngine, and MindSpider in the full repeat test.
- Strong Chinese synthesis and summarization behavior.

Weaknesses:

- Slower than `deepseek-chat`.
- Lower score than `deepseek-chat` for ReportEngine.

Best use:

- MediaEngine
- MindSpider
- ForumEngine if richer Chinese host output is preferred

### Bocha

Strengths:

- Strong scores for Chinese media/web retrieval.
- Stable success rate after credentials were corrected.
- Good fit for MediaEngine and QueryEngine retrieval.

Weaknesses:

- Average latency is high: about 13-14 seconds.
- MindSpider search coverage currently has fewer test cases than MediaEngine/QueryEngine.

Best use:

- Chinese media/web retrieval.
- Public-opinion discovery where richer retrieval quality is more important than immediate response speed.

Operational note:

- Add caching for repeated queries.
- Consider asynchronous/concurrent search execution.
- Keep timeout and retry policies conservative.

## Final Recommendation

### Quality-Oriented Configuration

```env
MEDIA_ENGINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEDIA_ENGINE_MODEL_NAME=qwen-plus

QUERY_ENGINE_BASE_URL=https://api.deepseek.com
QUERY_ENGINE_MODEL_NAME=deepseek-chat

REPORT_ENGINE_BASE_URL=https://api.deepseek.com
REPORT_ENGINE_MODEL_NAME=deepseek-chat

FORUM_HOST_BASE_URL=https://api.deepseek.com
FORUM_HOST_MODEL_NAME=deepseek-chat

MINDSPIDER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MINDSPIDER_MODEL_NAME=qwen-plus

SEARCH_TOOL_TYPE=BochaAPI
BOCHA_BASE_URL=https://api.bocha.cn/v1/ai-search
```

### Simple/Low-Latency Configuration

```env
MEDIA_ENGINE_BASE_URL=https://api.deepseek.com
MEDIA_ENGINE_MODEL_NAME=deepseek-chat

QUERY_ENGINE_BASE_URL=https://api.deepseek.com
QUERY_ENGINE_MODEL_NAME=deepseek-chat

REPORT_ENGINE_BASE_URL=https://api.deepseek.com
REPORT_ENGINE_MODEL_NAME=deepseek-chat

FORUM_HOST_BASE_URL=https://api.deepseek.com
FORUM_HOST_MODEL_NAME=deepseek-chat

MINDSPIDER_BASE_URL=https://api.deepseek.com
MINDSPIDER_MODEL_NAME=deepseek-chat

SEARCH_TOOL_TYPE=BochaAPI
BOCHA_BASE_URL=https://api.bocha.cn/v1/ai-search
```

## Recommended Next Steps

1. Manually review `results_full_r3\manual_review.csv`, especially:
   - `query_engine` JSON classification cases.
   - `report_engine` long-form and revision cases.
   - `media_engine` noise-filtering cases.
2. If final report quality is critical, run a second round with `qwen-max-compatible` enabled and compare only ReportEngine/MediaEngine cases.
3. Add real project outputs as additional test cases:
   - real MediaEngine source sets
   - real QueryEngine artifacts
   - real ReportEngine input JSON
4. Add cost values to `providers.local.json` once provider pricing is finalized.
5. Add caching/concurrency around Bocha before production use because search latency is the main operational risk.

## Artifacts

Primary files:

- `results_full_r3\summary.csv`
- `results_full_r3\recommendations.md`
- `results_full_r3\manual_review.csv`
- `results_full_r3\raw_results.jsonl`

Configuration files:

- `providers.local.json`
- `test_cases.json`
- `engine_profiles.json`
- `run_evaluation.py`
