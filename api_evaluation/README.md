# API Evaluation Toolkit

This directory contains a standalone evaluation harness and completed benchmark results for selecting APIs for:

- `MediaEngine`
- `QueryEngine`
- `ReportEngine`
- `ForumEngine`
- `MindSpider`

It is intentionally independent from the main application runtime. The goal is to test candidate LLM/search APIs before wiring them into the production engines.

## Current Evaluation Summary

The completed evaluation tested:

- `deepseek-chat`
- `deepseek-reasoner`
- `qwen-plus-compatible`
- `bocha` search

Two rounds were run:

- `results_smoke_real`: 1 repetition per test case.
- `results_full_r3`: 3 repetitions per test case.

The 3-run evaluation completed with no failed calls. All tested providers had `success_rate=1.0`.

Final pragmatic recommendation:

| Engine | Recommended API |
| --- | --- |
| MediaEngine LLM | `qwen-plus-compatible` |
| MediaEngine Search | `bocha` |
| QueryEngine LLM | `deepseek-chat` |
| QueryEngine Search | `bocha` |
| ReportEngine LLM | `deepseek-chat` |
| ForumEngine LLM | `deepseek-chat` or `qwen-plus-compatible` |
| MindSpider LLM | `qwen-plus-compatible` |
| MindSpider Search | `bocha` |

If operational simplicity and lower latency matter more than small quality gains, use `deepseek-chat` for all LLM engines and `bocha` for search.

## Key Results

Source: `results_full_r3/summary.csv`

| Engine | Type | Best Provider | Score | Avg Latency |
| --- | --- | --- | ---: | ---: |
| MediaEngine | LLM | `qwen-plus-compatible` | 4.418 | 5.673s |
| MediaEngine | Search | `bocha` | 4.646 | 14.399s |
| QueryEngine | LLM | `deepseek-chat` | 4.505 | 2.423s |
| QueryEngine | Search | `bocha` | 4.668 | 13.006s |
| ReportEngine | LLM | `deepseek-chat` | 4.227 | 5.479s |
| ForumEngine | LLM | `qwen-plus-compatible` | 4.269 | 3.235s |
| MindSpider | LLM | `qwen-plus-compatible` | 4.378 | 5.217s |
| MindSpider | Search | `bocha` | 4.709 | 13.840s |

Notable observations:

- `deepseek-chat` is the best practical default for QueryEngine and ReportEngine.
- `qwen-plus-compatible` is strongest for Chinese synthesis/topic extraction tasks.
- `deepseek-reasoner` did not outperform the other candidates enough to justify its higher latency in this test set.
- `bocha` search scored well but is relatively slow, with average latency around 13-14 seconds.

See the full report: [`API_EVALUATION_REPORT.md`](API_EVALUATION_REPORT.md).

## Directory Contents

| Path | Purpose |
| --- | --- |
| `run_evaluation.py` | Main benchmark runner. |
| `providers.example.json` | Safe provider config template. |
| `providers.local.json` | Current evaluated provider config. Uses environment variable names only; no API keys are stored. |
| `test_cases.json` | LLM and search test cases mapped to engine profiles. |
| `engine_profiles.json` | Engine-specific scoring weights. |
| `EVALUATION_RUNBOOK.md` | Step-by-step run instructions. |
| `API_EVALUATION_REPORT.md` | Final evaluation report and recommendations. |
| `results_smoke_real/` | Successful 1-run smoke test summaries. |
| `results_full_r3/` | Successful 3-run evaluation summaries. |

Large raw outputs are intentionally ignored:

- `raw_results.jsonl`
- `__pycache__/`
- dry-run/config-check result folders

## How to Reproduce

Set the required keys in the current PowerShell session:

```powershell
cd D:\huang\Desktop\Project\api_evaluation

$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:DASHSCOPE_API_KEY="your_qwen_key"
$env:BOCHA_WEB_SEARCH_API_KEY="your_bocha_key"
```

Run a quick smoke test:

```powershell
python run_evaluation.py --providers providers.local.json --out results_smoke_real
```

Run the more reliable 3-run evaluation:

```powershell
python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

Review outputs:

```powershell
notepad results_full_r3\summary.csv
notepad results_full_r3\recommendations.md
notepad results_full_r3\manual_review.csv
```

If only the summary files need to be rebuilt from existing raw results, use:

```powershell
python run_evaluation.py --providers providers.local.json --out results_full_r3 --summarize-only
```

## Scoring Method

Each provider is scored on a 0-5 scale. Engine-level weighted scores are computed from metrics in `engine_profiles.json`.

LLM metrics include:

- `task_success`
- `format_compliance`
- `grounding`
- `reasoning_quality`
- `language_quality`
- `latency`
- `cost`
- `stability`

Search metrics include:

- `relevance`
- `freshness`
- `source_quality`
- `coverage`
- `metadata_quality`
- `latency`
- `cost`
- `parseability`

The script also checks:

- required JSON fields
- required keywords
- forbidden hallucination-prone phrases
- output length constraints
- API latency and call failures

Automatic scoring is useful for screening, but final production selection should include manual review of `manual_review.csv`, especially for:

- QueryEngine JSON classification and gap filling.
- ReportEngine long-form report quality.
- MediaEngine rumor/noise filtering.
- Bocha result relevance and source diversity.

## Current Deployment Suggestions

Quality-oriented setup:

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

Simple low-latency setup:

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

## Extending the Evaluation

To add another LLM provider:

1. Add it to `providers.local.json`.
2. Use an environment variable for the key.
3. Set `"enabled": true`.
4. Rerun the benchmark.

To add another search provider:

1. Add the provider config.
2. Implement a `call_*` adapter in `run_evaluation.py`.
3. Normalize its response to `{answer, results, images, cards}` where possible.
4. Add test cases if the provider has special capabilities.

Recommended future comparisons:

- Enable `qwen-max-compatible` for a second-round quality test.
- Compare `bocha` with Anspire for Chinese search latency/relevance tradeoffs.
- Add real project artifacts as test cases from MediaEngine, QueryEngine, and ReportEngine outputs.
- Add actual provider pricing in `providers.local.json` so cost scoring becomes meaningful.
