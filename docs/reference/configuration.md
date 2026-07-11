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
| Evaluated default | `deepseek-chat` | `qwen-plus-compatible` / `qwen-plus` | `deepseek-chat` | `qwen-plus-compatible` / `qwen-plus` | `TavilyAPI` |
| Low-latency simplification | `deepseek-chat` | `deepseek-chat` | `deepseek-chat` | `deepseek-chat` | `TavilyAPI` |
| Strong-report profile | `deepseek-chat` | `qwen-plus` | A stronger OpenAI-compatible report model | `qwen-plus` | `TavilyAPI` |

The report engine is sensitive to structured-output quality. If chart blocks, table blocks, or long-form report sections degrade, keep the same API shape but try a stronger report model and rerun the report task.

## Search Providers

| Key | Default | Purpose |
| --- | --- | --- |
| `SEARCH_TOOL_TYPE` | `TavilyAPI` | Selects `TavilyAPI`, `BochaAPI`, or `AnspireAPI`. |
| `TAVILY_API_KEY` | empty | Tavily search key when the Tavily path is selected. |
| `ANSPIRE_BASE_URL` | `https://plugin.anspire.cn/api/ntsearch/search` | Anspire endpoint. |
| `ANSPIRE_API_KEY` | empty | Anspire key. |
| `BOCHA_BASE_URL` | `https://api.bocha.cn/v1/ai-search` | Bocha endpoint. |
| `BOCHA_WEB_SEARCH_API_KEY` | empty | Bocha key. |

Search providers acquire external evidence. They are separate from semantic quality providers. If the selected provider key is blank, the artifact records `not_configured` and the Coordinator continues. If a configured provider fails, the artifact records `provider:error`; it does not silently pretend search succeeded.

## Coordinator Semantic Providers

| Key | Default | Purpose |
| --- | --- | --- |
| `JINA_API_KEY` | empty | Primary semantic provider key. Enables Jina embeddings for semantic duplicate clustering and Jina rerank for relevance scoring. |
| `JINA_EMBEDDING_BASE_URL` | `https://api.jina.ai/v1/embeddings` | Jina embeddings endpoint. |
| `JINA_EMBEDDING_MODEL` | `jina-embeddings-v5-text-small` | Text embedding model for semantic duplicate detection. |
| `JINA_EMBEDDING_DIMENSIONS` | empty | Optional dimensions override. Leave blank for provider default. |
| `JINA_RERANK_BASE_URL` | `https://api.jina.ai/v1/rerank` | Jina rerank endpoint. |
| `JINA_RERANK_MODEL` | `jina-reranker-v3` | Jina rerank model for relevance scoring. |
| `COORDINATOR_MAX_EMBEDDING_ITEMS` | `120` | Max items sent to embedding per run. |
| `COORDINATOR_MAX_RERANK_DOCUMENTS` | `40` | Max documents sent to rerank per run. |
| `COORDINATOR_PROVIDER_TIMEOUT` | `30` | Timeout seconds for semantic provider calls. |
| `COORDINATOR_SEMANTIC_DUPLICATE_THRESHOLD` | `0.92` | Cosine threshold for embedding-assisted duplicate clustering. |

Provider routing:

```text
Jina configured -> use Jina embeddings + Jina rerank.
Jina missing -> deterministic duplicate and relevance rules remain active.
No semantic key -> deterministic hash/rule route remains active and diagnostics show not_configured.
```

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
| `QUERY_MAX_SEARCH_ITERATIONS` | `2` | QueryEngine graph gap-fill search rounds. Active Coordinator follow-up rounds use `COORDINATOR_MAX_RESEARCH_ROUNDS`. |
| `SEARCH_TIMEOUT` | `60` | Single search timeout for web source acquisition. |
| `SEARCH_CONTENT_MAX_LENGTH` | `50000` | Snippet length passed to LLM prompts. |
| `TAVILY_SEARCH_MAX_CONCURRENT` | `3` | QueryEngine graph Tavily concurrency. Active Coordinator source gateway uses bounded per-task query budgets. |
| `LLM_SHORT_TASK_TIMEOUT` | `120` | Short LLM request timeout. |
| `LLM_LONG_TASK_TIMEOUT` | `600` | Streaming/long LLM request timeout. |
| `LLM_STREAM_IDLE_TIMEOUT` | `240` | Stream idle watchdog window. |
| `MEDIA_USE_LLM_REPORT_FORMAT` | `False` | Uses direct MediaEngine report assembly for throughput. |
| `MEDIA_SEARCH_HTTP_TIMEOUT` | `60` | MediaEngine search HTTP timeout. |
| `COORDINATOR_MEDIA_AGENT_TIMEOUT` | `10800` | Parent-graph deadline for each Media specialist task. |
| `COORDINATOR_QUERY_AGENT_TIMEOUT` | `1800` | Parent-graph deadline for each Query specialist task. |
| `COORDINATOR_MAX_RESEARCH_ROUNDS` | `1` | Maximum global sufficiency/follow-up rounds after initial Query/Media fan-out. |
| `COORDINATOR_ENABLE_MINDSPIDER_DB` | `False` | Enables read-only QueryEngine searches over existing MindSpider crawl tables. |
| `COORDINATOR_ENABLE_MEDIA_AGENT` | `True` | Runs MediaEngine; disable only for an explicit Query/MindSpider-only run. |
| `COORDINATOR_ALLOW_MINDSPIDER_CRAWL_TRIGGER` | `False` | Separately permits stale/missing-data enrichment to start a crawl subprocess. Keep false for analysis-only servers. |
| `COORDINATOR_QUERY_MAX_SOURCES` | `120` | Caps the combined web and MindSpider evidence accepted from the primary QueryAgent task. |

## MediaEngine Cache And Performance

| Mechanism | Purpose |
| --- | --- |
| `AgentCoordinator/cache/media_agent_<hash>.md` | Reuses MediaEngine Markdown output for matching Coordinator topics. |
| Parallel paragraph processing | Processes MediaEngine report paragraphs concurrently when `MEDIA_PARAGRAPH_WORKERS > 1`. |
| Stream-idle watchdog | Stops stalled streaming reads according to `LLM_STREAM_IDLE_TIMEOUT`. |
| Timeout budgets | Keeps legacy QueryEngine/MediaEngine and active ReportEngine request behavior explicit and tunable. |

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

JINA_API_KEY=your_jina_key
JINA_EMBEDDING_BASE_URL=https://api.jina.ai/v1/embeddings
JINA_EMBEDDING_MODEL=jina-embeddings-v5-text-small
JINA_RERANK_BASE_URL=https://api.jina.ai/v1/rerank
JINA_RERANK_MODEL=jina-reranker-v3
SEARCH_TOOL_TYPE=TavilyAPI
TAVILY_API_KEY=your_tavily_key

REPORT_OUTPUT_LANGUAGE=en
```

Provider recommendations are documented in [API Evaluation](../quality/api-evaluation.md).

`COORDINATOR_ALLOW_REPLAY_FALLBACK` is retained for the previous intelligence-layer test/baseline path. The active Query/Media fusion graph does not synthesize replay fixtures.
