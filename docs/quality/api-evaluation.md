# API Evaluation

`api_evaluation/` is a standalone harness for comparing LLM and search providers before wiring them into the runtime configuration.

## Purpose

| Goal | Explanation |
| --- | --- |
| Select providers by evidence | Compare quality, latency, stability, format compliance, and cost proxy. |
| Keep runtime independent | Evaluation scripts do not run as part of Flask or Signal Studio. |
| Support manual review | Outputs include `manual_review.csv` for qualitative inspection. |
| Preserve repeatability | Test cases, provider profiles, and score summaries are stored with the harness. |

## Evaluated Engines

| Engine | Provider Type |
| --- | --- |
| MediaEngine | LLM and search |
| QueryEngine | LLM and search |
| ReportEngine | LLM |
| ForumEngine | LLM |
| MindSpider | LLM and search |

## Candidate Providers

The completed selection round evaluated these enabled providers:

| Provider | Type | Runtime Target |
| --- | --- | --- |
| `deepseek-chat` | LLM | Fast default for planning, classification, reporting, and forum hosting. |
| `deepseek-reasoner` | LLM | Reasoning candidate for difficult structured tasks. |
| `qwen-plus-compatible` | LLM | Chinese synthesis, topic extraction, and clustering candidate. |
| `bocha` | Search | Chinese media and web retrieval candidate. |

The harness also defines optional profiles for `qwen-max-compatible`, Kimi, Gemini-compatible access, Tavily, and Anspire. They can be enabled in `api_evaluation/providers.local.json` for targeted comparison rounds.

## Current Results Summary

The completed benchmark includes:

| Result Folder | Description |
| --- | --- |
| `api_evaluation/results_smoke_real/` | One repetition per test case. |
| `api_evaluation/results_full_r3/` | Three repetitions per test case. |

The current default recommendations are based on `api_evaluation/results_full_r3/summary.csv`:

| Engine | Provider Type | Recommended API | Score | Success Rate | Avg Latency |
| --- | --- | --- | ---: | ---: | ---: |
| MediaEngine | LLM | `qwen-plus-compatible` | 4.418 | 1.00 | 5.673s |
| MediaEngine | Search | `bocha` | 4.646 | 1.00 | 14.399s |
| QueryEngine | LLM | `deepseek-chat` | 4.505 | 1.00 | 2.423s |
| QueryEngine | Search | `bocha` | 4.668 | 1.00 | 13.006s |
| ReportEngine | LLM | `deepseek-chat` | 4.227 | 1.00 | 5.479s |
| ForumEngine | LLM | `qwen-plus-compatible` or `deepseek-chat` | 4.269 / 4.238 | 1.00 | 3.235s / 1.896s |
| MindSpider | LLM | `qwen-plus-compatible` | 4.378 | 1.00 | 5.217s |
| MindSpider | Search | `bocha` | 4.709 | 1.00 | 13.840s |

For simpler low-latency operation, the existing recommendation is `deepseek-chat` for all LLM engines and `bocha` for search.

For the broader evidence summary, see [Evidence Dashboard](evidence-dashboard.md).

## Pragmatic Assignment

| Runtime Area | Default API |
| --- | --- |
| MediaEngine | `qwen-plus-compatible` |
| QueryEngine | `deepseek-chat` |
| ReportEngine | `deepseek-chat` |
| ForumEngine | `deepseek-chat` |
| MindSpider | `qwen-plus-compatible` |
| Search | `bocha` |

## Key Observations

| Observation | Interpretation |
| --- | --- |
| All tested providers reached `success_rate=1.00` in the full repeat run. | The evaluated profile is stable enough for runtime configuration. |
| `deepseek-chat` led QueryEngine and ReportEngine. | Strong default for structured evidence planning and report generation. |
| `qwen-plus-compatible` led MediaEngine, MindSpider, and ForumEngine. | Strong fit for Chinese synthesis, topic extraction, clustering, and host language quality. |
| ForumEngine had a small score spread. | `deepseek-chat` is the practical default when latency matters. |
| `deepseek-reasoner` did not win an engine group. | Higher latency was not justified by this benchmark. |
| `bocha` search scored strongly but averaged about 13-14 seconds. | Good retrieval quality; tune timeout/concurrency around it. |

## Full Result Matrix

Source: `api_evaluation/results_full_r3/summary.csv`.

| Engine | Type | Provider | Score | Success Rate | Avg Latency | Cases |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ForumEngine | LLM | `deepseek-chat` | 4.238 | 1.00 | 1.896s | 6 |
| ForumEngine | LLM | `deepseek-reasoner` | 4.224 | 1.00 | 3.175s | 6 |
| ForumEngine | LLM | `qwen-plus-compatible` | 4.269 | 1.00 | 3.235s | 6 |
| MediaEngine | LLM | `deepseek-chat` | 4.407 | 1.00 | 3.196s | 6 |
| MediaEngine | LLM | `deepseek-reasoner` | 4.351 | 1.00 | 5.068s | 6 |
| MediaEngine | LLM | `qwen-plus-compatible` | 4.418 | 1.00 | 5.673s | 6 |
| MediaEngine | Search | `bocha` | 4.646 | 1.00 | 14.399s | 18 |
| MindSpider | LLM | `deepseek-chat` | 4.255 | 1.00 | 1.993s | 6 |
| MindSpider | LLM | `deepseek-reasoner` | 4.243 | 1.00 | 3.991s | 6 |
| MindSpider | LLM | `qwen-plus-compatible` | 4.378 | 1.00 | 5.217s | 6 |
| MindSpider | Search | `bocha` | 4.709 | 1.00 | 13.840s | 3 |
| QueryEngine | LLM | `deepseek-chat` | 4.505 | 1.00 | 2.423s | 9 |
| QueryEngine | LLM | `deepseek-reasoner` | 4.399 | 1.00 | 5.612s | 9 |
| QueryEngine | LLM | `qwen-plus-compatible` | 4.466 | 1.00 | 6.286s | 9 |
| QueryEngine | Search | `bocha` | 4.668 | 1.00 | 13.006s | 18 |
| ReportEngine | LLM | `deepseek-chat` | 4.227 | 1.00 | 5.479s | 9 |
| ReportEngine | LLM | `deepseek-reasoner` | 4.196 | 1.00 | 9.155s | 9 |
| ReportEngine | LLM | `qwen-plus-compatible` | 4.167 | 1.00 | 12.373s | 9 |

## Smoke And Repeat Comparison

The one-run smoke test and the three-run repeat test produced the same provider direction for MediaEngine, QueryEngine, ReportEngine, MindSpider, and search. ForumEngine moved from a reasoning-model lead in the smoke run to a Qwen Plus lead in the repeat run, with a small score spread. The runtime assignment therefore favors `deepseek-chat` for ForumEngine when fast response is more important than small language-style gains.

## Provider Decision Notes

| Provider | Strongest Fit | Decision Note |
| --- | --- | --- |
| `deepseek-chat` | QueryEngine, ReportEngine, ForumEngine | Best practical default when speed and broad task coverage matter. |
| `deepseek-reasoner` | Optional hard-case reasoning | Stable candidate, but not the default winner in the repeat run. |
| `qwen-plus-compatible` | MediaEngine, MindSpider | Best score for Chinese synthesis, extraction, and clustering tasks. |
| `bocha` | Search | Strong retrieval quality for Chinese media and web evidence. |

## Test Coverage

| Engine | Cases Covered |
| --- | --- |
| MediaEngine | Chinese public-opinion synthesis, rumor/noise filtering, evidence boundary control. |
| QueryEngine | Stance classification, retrieval planning, gap filling, no-hallucination behavior. |
| ReportEngine | Long-form report generation, chart/table planning, conservative revision. |
| ForumEngine | Multi-agent discussion moderation and concise intervention. |
| MindSpider | Keyword extraction, topic clustering, duplicate topic merge. |
| Search | Chinese media/web retrieval, policy retrieval, rumor/regulatory response, brand recall, platform complaints, macro signals, English context, disaster response. |

## Harness Files

| Path | Purpose |
| --- | --- |
| `api_evaluation/run_evaluation.py` | Main runner and scoring implementation. |
| `api_evaluation/providers.example.json` | Safe provider config template. |
| `api_evaluation/providers.local.json` | Local provider config using environment variable names. |
| `api_evaluation/test_cases.json` | Test cases by engine. |
| `api_evaluation/engine_profiles.json` | Engine-specific scoring weights. |
| `api_evaluation/results_*/summary.csv` | Score summary. |
| `api_evaluation/results_*/manual_review.csv` | Manual review worksheet. |
| `api_evaluation/results_*/recommendations.md` | Generated recommendation output when the runner writes a full artifact set. |
| `api_evaluation/results_*/raw_results.jsonl` | Raw call outputs generated by the runner and ignored by Git. |

Generated JSONL outputs, bytecode caches, and intermediate result folders are excluded by `api_evaluation/.gitignore`. The committed evidence focuses on the stable CSV review artifacts and this documentation page.

## Run Commands

Use `uv` from `api_evaluation/`:

```powershell
cd api_evaluation
$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:DASHSCOPE_API_KEY="your_qwen_key"
$env:BOCHA_WEB_SEARCH_API_KEY="your_bocha_key"
uv run --python 3.11 --with-requirements ../requirements.txt python run_evaluation.py --providers providers.local.json --out results_smoke_real
```

Run the three-run evaluation:

```powershell
uv run --python 3.11 --with-requirements ../requirements.txt python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

Rebuild summaries from existing raw results:

```powershell
uv run --python 3.11 --with-requirements ../requirements.txt python run_evaluation.py --providers providers.local.json --out results_full_r3 --summarize-only
```

Conda equivalent:

```powershell
conda activate <project-env>
cd api_evaluation
python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

If optional providers are enabled, also set their key variables: `MOONSHOT_API_KEY`, `AIHUBMIX_API_KEY`, `TAVILY_API_KEY`, or `ANSPIRE_API_KEY`.

Review the generated artifacts:

```powershell
Get-Content results_full_r3\summary.csv
Import-Csv results_full_r3\manual_review.csv | Format-Table -AutoSize
Get-Content results_full_r3\recommendations.md
```

## Scoring Dimensions

| LLM Metrics | Search Metrics |
| --- | --- |
| `task_success` | `relevance` |
| `format_compliance` | `freshness` |
| `grounding` | `source_quality` |
| `reasoning_quality` | `coverage` |
| `language_quality` | `metadata_quality` |
| `latency` | `latency` |
| `cost` | `cost` |
| `stability` | `parseability` |

The runner also checks required JSON fields, required keywords, forbidden hallucination-prone phrases, output length constraints, API latency, and call success.

## Manual Review Checklist

Automatic scoring is used for screening. For final provider selection, inspect `manual_review.csv` and focus on:

| Area | What To Check |
| --- | --- |
| QueryEngine JSON cases | Valid JSON, stance accuracy, gap filling, source-grounded claims. |
| ReportEngine long-form cases | Structure, conservative wording, chart/table planning, revision quality. |
| MediaEngine conflict/noise cases | Rumor handling, source boundaries, useful synthesis. |
| Bocha search rows | Relevance, source diversity, freshness, metadata quality. |

## Applying Results To Runtime

Update `.env` or the Settings drawer with the selected provider:

```env
QUERY_ENGINE_BASE_URL=https://api.deepseek.com
QUERY_ENGINE_MODEL_NAME=deepseek-chat

REPORT_ENGINE_BASE_URL=https://api.deepseek.com
REPORT_ENGINE_MODEL_NAME=deepseek-chat

MEDIA_ENGINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEDIA_ENGINE_MODEL_NAME=qwen-plus

SEARCH_TOOL_TYPE=TavilyAPI
TAVILY_API_KEY=your_tavily_key
```

Quality-oriented profile:

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

SEARCH_TOOL_TYPE=TavilyAPI
TAVILY_API_KEY=your_tavily_key
```

Simple low-latency profile:

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

SEARCH_TOOL_TYPE=TavilyAPI
TAVILY_API_KEY=your_tavily_key
```

## Extending The Evaluation

| Change | Action |
| --- | --- |
| Add an LLM provider | Add it to `providers.local.json`, use an environment-variable key, set `enabled=true`, rerun the benchmark. |
| Add a search provider | Add provider config, implement a `call_*` adapter in `run_evaluation.py`, normalize to `{answer, results, images, cards}`. |
| Test stronger finalists | Enable `qwen-max-compatible`, Kimi, or Gemini-compatible profiles for targeted second-round runs. |
| Compare search alternatives | Enable Bocha or Anspire and rerun retrieval cases against the Tavily profile. |
| Improve cost scoring | Add actual provider pricing to `providers.local.json`. |
| Increase realism | Add real MediaEngine, QueryEngine, and ReportEngine artifacts as test cases. |

See [Configuration](../reference/configuration.md) for full settings.
