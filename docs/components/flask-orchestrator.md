# Flask Orchestrator

`app.py` is the runtime gateway for Signal Studio, AgentCoordinator, ReportEngine, configuration, legacy process controls, ForumEngine logs, and observability.

## Responsibilities

| Responsibility | Implementation |
| --- | --- |
| Serve Signal Studio | `GET /` renders `templates/index.html`. |
| Register ReportEngine APIs | `app.register_blueprint(report_bp, url_prefix='/api/report')`. |
| Manage runtime status | `/api/system/status`, `/api/system/start`, `/api/system/shutdown`. |
| Run Coordinator tasks | `/api/coordinator/run`, background thread, task registry. |
| Expose latest analysis artifact | `/api/coordinator/latest`. |
| Store feedback | `/api/coordinator/feedback`. |
| Edit configuration | `/api/config`. |
| Load traces | `/api/observability/langsmith`. |
| Preserve legacy controls | `/api/start/{app_name}`, `/api/stop/{app_name}`, `/api/forum/*`, Socket.IO. |

## Signal Studio Startup

`initialize_system_components()` prepares the final Signal Studio runtime:

| Step | Behavior |
| --- | --- |
| Stop Streamlit apps | Iterates over `STREAMLIT_SCRIPTS` and stops compatibility Query/Media apps. |
| Stop ForumEngine | Stops the monitor and marks `forum` as stopped. |
| Initialize ReportEngine | Calls `initialize_report_engine()` from the ReportEngine Blueprint. |
| Return readiness | Reports readiness and diagnostics to `/api/system/start`. |

## Coordinator Task Model

| Field | Meaning |
| --- | --- |
| `task_id` | `coord_<timestamp>_<suffix>`. |
| `query` | Analysis topic. |
| `has_feedback` | Whether refinement feedback was passed. |
| `status` | `queued`, `running`, `completed`, or `error`. |
| `progress` | UI progress percentage mapped by node. |
| `message` | Human-readable status. |
| `created_at`, `updated_at` | UTC timestamps. |
| `thread_id` | Coordinator checkpoint thread id when available. |
| `error` | Terminal diagnostic message. |

## Configuration Editing

`/api/config` reads and writes selected keys in `config.py`. The allowlist is `CONFIG_KEYS` in `app.py`.

| Group | Example Keys |
| --- | --- |
| Server | `HOST`, `PORT` |
| Engine LLMs | `QUERY_ENGINE_API_KEY`, `MEDIA_ENGINE_MODEL_NAME`, `REPORT_ENGINE_BASE_URL` |
| Forum and keyword optimizer | `FORUM_HOST_*`, `KEYWORD_OPTIMIZER_*` |
| Search | `SEARCH_TOOL_TYPE`, `TAVILY_API_KEY`, `BOCHA_WEB_SEARCH_API_KEY`, `ANSPIRE_API_KEY` |
| Observability | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` |
| Coordinator timeouts | `COORDINATOR_MEDIA_AGENT_TIMEOUT`, `COORDINATOR_QUERY_AGENT_TIMEOUT` |

See [Configuration](../reference/configuration.md).

## Legacy Process Controls

The orchestrator still tracks legacy app processes:

| Legacy App | Typical Port | Notes |
| --- | ---: | --- |
| `media` | 8502 | Streamlit MediaEngine app. |
| `query` | 8503 | Streamlit QueryEngine app. |
| `forum` | N/A | ForumEngine monitor/log surface. |

These endpoints are kept for compatibility and diagnostics. The final Signal Studio path does not require them.

## Related Documents

- [Runtime Flow](../architecture/runtime-flow.md)
- [API Reference](../reference/api.md)
- [Signal Studio](signal-studio.md)
