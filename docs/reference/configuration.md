# Configuration

Configuration is centralized in `config.py::Settings` and can be loaded from `.env`, environment variables, or selected runtime edits through `/api/config`.

## Configuration Sources

| Source | Usage |
| --- | --- |
| `.env` | Local runtime settings and secrets. |
| `.env.example` | Safe template. |
| Environment variables | Override settings without editing files. |
| `config.py` defaults | Pydantic Settings defaults. |
| `/api/config` | Edits selected allowlisted keys in `config.py`. |

## Server And Runtime

| Key | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Flask bind host. |
| `PORT` | `5000` | Flask port. |
| `OUTPUT_DIR` | `output` | General output directory. |
| `SAVE_INTERMEDIATE_STATES` | `False` | Save intermediate Query/Media state JSON. |

## ReportEngine Paths

| Key | Default | Purpose |
| --- | --- | --- |
| `LOG_FILE` | `logs/report.log` | ReportEngine log sink. |
| `CHAPTER_OUTPUT_DIR` | `output/chapters` | Generated chapter JSON. |
| `DOCUMENT_IR_OUTPUT_DIR` | `output/document_ir` | Document IR output. |
| `TEMPLATE_DIR` | `ReportEngine/report_template` | Runtime report templates. |
| `JSON_ERROR_LOG_DIR` | `output/json_error_logs` | JSON parsing/repair diagnostics. |
| `CHAPTER_JSON_MAX_ATTEMPTS` | `5` | Max attempts for chapter JSON generation/repair. |

## Output Language

| Key | Default | Purpose |
| --- | --- | --- |
| `REPORT_OUTPUT_LANGUAGE` | `en` | `en` enforces English report prose/headings; `zh` allows Chinese. |
| `REPORT_TRANSLATE_INPUT_TO_EN` | `True` | Translate Chinese upstream inputs before English report generation. |
| `REPORT_INPUT_TRANSLATION_TIMEOUT_SECONDS` | `45` | Max translation time before continuing. |

## LLM Providers

| Component | Keys |
| --- | --- |
| Insight Agent | `INSIGHT_ENGINE_API_KEY`, `INSIGHT_ENGINE_BASE_URL`, `INSIGHT_ENGINE_MODEL_NAME` |
| MediaEngine | `MEDIA_ENGINE_API_KEY`, `MEDIA_ENGINE_BASE_URL`, `MEDIA_ENGINE_MODEL_NAME` |
| QueryEngine | `QUERY_ENGINE_API_KEY`, `QUERY_ENGINE_BASE_URL`, `QUERY_ENGINE_MODEL_NAME` |
| ReportEngine | `REPORT_ENGINE_API_KEY`, `REPORT_ENGINE_BASE_URL`, `REPORT_ENGINE_MODEL_NAME` |
| MindSpider | `MINDSPIDER_API_KEY`, `MINDSPIDER_BASE_URL`, `MINDSPIDER_MODEL_NAME` |
| Forum Host | `FORUM_HOST_API_KEY`, `FORUM_HOST_BASE_URL`, `FORUM_HOST_MODEL_NAME` |
| Keyword Optimizer | `KEYWORD_OPTIMIZER_API_KEY`, `KEYWORD_OPTIMIZER_BASE_URL`, `KEYWORD_OPTIMIZER_MODEL_NAME` |

All main LLM clients are expected to work with OpenAI-compatible API shapes when configured with the correct key, base URL, and model name.

## Recommended Provider Profiles

Use one of these profiles before experimenting with other providers. The default profile is aligned with the stored `api_evaluation/results_full_r3` benchmark.

| Profile | QueryEngine | MediaEngine | ReportEngine | MindSpider | Search |
| --- | --- | --- | --- | --- | --- |
| Evaluated default | `deepseek-chat` | `qwen-plus-compatible` / `qwen-plus` | `deepseek-chat` | `qwen-plus-compatible` / `qwen-plus` | `BochaAPI` |
| Low-latency simplification | `deepseek-chat` | `deepseek-chat` | `deepseek-chat` | `deepseek-chat` | `BochaAPI` |
| Strong-report profile | `deepseek-chat` | `qwen-plus` | A stronger OpenAI-compatible report model | `qwen-plus` | `BochaAPI` |

The report engine is sensitive to structured-output quality. If chart blocks, table blocks, or long-form report sections degrade, keep the same API shape but try a stronger report model and rerun the report task.

## Search Providers

| Key | Default | Purpose |
| --- | --- | --- |
| `SEARCH_TOOL_TYPE` | `AnspireAPI` | Selects `AnspireAPI` or `BochaAPI`. |
| `TAVILY_API_KEY` | empty | Tavily search key when the Tavily path is selected. |
| `ANSPIRE_BASE_URL` | `https://plugin.anspire.cn/api/ntsearch/search` | Anspire endpoint. |
| `ANSPIRE_API_KEY` | empty | Anspire key. |
| `BOCHA_BASE_URL` | `https://api.bocha.cn/v1/ai-search` | Bocha endpoint. |
| `BOCHA_WEB_SEARCH_API_KEY` | empty | Bocha key. |

## Search And Coordinator Limits

| Key | Default | Purpose |
| --- | --- | --- |
| `DEFAULT_SEARCH_HOT_CONTENT_LIMIT` | `100` | Default hot-content count. |
| `DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE` | `50` | Topic limit per table. |
| `DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE` | `100` | Date-filtered topic limit. |
| `DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT` | `500` | Comment retrieval limit. |
| `DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT` | `200` | Platform search limit. |
| `MAX_REFLECTIONS` | `3` | Reflection iterations. |
| `MAX_PARAGRAPHS` | `6` | Max paragraph count. |
| `MEDIA_PARAGRAPH_WORKERS` | `3` | Parallel MediaEngine paragraph workers. |
| `MEDIA_PARAGRAPH_RETRY_PASSES` | `1` | Sequential recovery passes after parallel paragraph processing. |
| `MEDIA_REFLECTION_STATE_MAX_CHARS` | `50000` | Reflection-summary context cap for MediaEngine prompts. |
| `QUERY_MAX_SEARCH_ITERATIONS` | `2` | QueryEngine gap-fill search rounds. |
| `SEARCH_TIMEOUT` | `60` | Single search timeout. |
| `SEARCH_CONTENT_MAX_LENGTH` | `50000` | Snippet length passed to LLM prompts. |
| `TAVILY_SEARCH_MAX_CONCURRENT` | `3` | Max parallel subqueries. |
| `LLM_SHORT_TASK_TIMEOUT` | `120` | Short LLM request timeout. |
| `LLM_LONG_TASK_TIMEOUT` | `600` | Streaming/long LLM request timeout. |
| `LLM_STREAM_IDLE_TIMEOUT` | `240` | Stream idle watchdog window. |
| `MEDIA_USE_LLM_REPORT_FORMAT` | `False` | Uses direct MediaEngine report assembly for throughput. |
| `MEDIA_SEARCH_HTTP_TIMEOUT` | `60` | MediaEngine search HTTP timeout. |
| `COORDINATOR_MEDIA_AGENT_TIMEOUT` | `10800` | MediaEngine timeout inside Coordinator. |
| `COORDINATOR_QUERY_AGENT_TIMEOUT` | `1800` | QueryEngine timeout inside Coordinator. |

## MediaEngine Cache And Performance

| Mechanism | Purpose |
| --- | --- |
| `AgentCoordinator/cache/media_agent_<hash>.md` | Reuses MediaEngine Markdown output for matching Coordinator topics. |
| Parallel paragraph processing | Processes MediaEngine report paragraphs concurrently when `MEDIA_PARAGRAPH_WORKERS > 1`. |
| Stream-idle watchdog | Stops stalled streaming reads according to `LLM_STREAM_IDLE_TIMEOUT`. |
| Timeout budgets | Keeps QueryEngine, MediaEngine, and ReportEngine request behavior explicit and tunable. |

## Observability

| Key | Default | Purpose |
| --- | --- | --- |
| `LANGSMITH_TRACING` | `False` | Enables LangSmith/LangChain tracing. |
| `LANGSMITH_API_KEY` | empty | LangSmith API key. |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith endpoint. |
| `LANGSMITH_PROJECT` | `public-opinion-analysis` | Project name. |
| `LANGCHAIN_TRACING_V2` | empty | Backward-compatible tracing flag. |
| `LANGCHAIN_PROJECT` | empty | Backward-compatible project name. |

## Sensitive Input Filter

| Key | Default | Purpose |
| --- | --- | --- |
| `ENABLE_SENSITIVE_INPUT_FILTER` | `True` | Blocks user-supplied text matching configured sensitive words. |
| `SENSITIVE_WORDS_FILE` | `config/sensitive_words.txt` | One blocked term per line; `#` comments supported. |

Sensitive input checks are used before Coordinator runs and report generation.

## Example Minimal `.env`

```env
HOST=0.0.0.0
PORT=5000

QUERY_ENGINE_API_KEY=your_query_key
QUERY_ENGINE_BASE_URL=https://api.deepseek.com
QUERY_ENGINE_MODEL_NAME=deepseek-chat

MEDIA_ENGINE_API_KEY=your_media_key
MEDIA_ENGINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEDIA_ENGINE_MODEL_NAME=qwen-plus

REPORT_ENGINE_API_KEY=your_report_key
REPORT_ENGINE_BASE_URL=https://api.deepseek.com
REPORT_ENGINE_MODEL_NAME=deepseek-chat

SEARCH_TOOL_TYPE=BochaAPI
BOCHA_WEB_SEARCH_API_KEY=your_bocha_key
BOCHA_BASE_URL=https://api.bocha.cn/v1/ai-search

REPORT_OUTPUT_LANGUAGE=en
```

Provider recommendations are documented in [API Evaluation](../quality/api-evaluation.md).
