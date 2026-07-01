# API Reference

This reference covers the HTTP APIs implemented by `app.py` and `ReportEngine/flask_interface.py`.

Base URL for local source runs:

```text
http://127.0.0.1:5000
```

Most endpoints return JSON with a `success` field unless they stream events or return files.

## End-To-End API Example

This sequence mirrors the main Signal Studio workflow. It assumes Flask is already running at `http://127.0.0.1:5000`.

Start the final runtime:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/system/start
```

Start an analysis task:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/coordinator/run `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Public reaction to a new AI policy\"}"
```

Example response:

```json
{
  "success": true,
  "task": {
    "task_id": "coord_20260701_120000_ab12cd",
    "query": "Public reaction to a new AI policy",
    "has_feedback": false,
    "status": "queued",
    "progress": 0,
    "message": "Coordinator task queued"
  }
}
```

Poll the task:

```powershell
curl.exe http://127.0.0.1:5000/api/coordinator/task/coord_20260701_120000_ab12cd
```

Load the latest artifact after completion:

```powershell
curl.exe http://127.0.0.1:5000/api/coordinator/latest
```

Start report generation:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/report/generate `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Public reaction to a new AI policy\"}"
```

Stream report progress:

```powershell
curl.exe -N http://127.0.0.1:5000/api/report/stream/report_20260701_120500_ab12cd
```

Export once completed:

```powershell
curl.exe -OJ http://127.0.0.1:5000/api/report/download/report_20260701_120500_ab12cd
curl.exe -OJ http://127.0.0.1:5000/api/report/export/md/report_20260701_120500_ab12cd
curl.exe -OJ http://127.0.0.1:5000/api/report/export/pdf/report_20260701_120500_ab12cd
```

## Signal Studio And Runtime

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Render Signal Studio shell. |
| `GET` | `/api/system/status` | Return final Signal Studio runtime status. |
| `POST` | `/api/system/start` | Start final Signal Studio runtime components. |
| `POST` | `/api/system/shutdown` | Stop child processes and schedule server shutdown. |
| `GET` | `/api/config` | Read selected configuration values. |
| `POST` | `/api/config` | Persist selected configuration values to `config.py`. |
| `GET` | `/api/observability/langsmith` | Load recent LangSmith traces or local trace status. |

### `GET /api/system/status`

Response:

| Field | Type | Description |
| --- | --- | --- |
| `success` | boolean | Always true on normal response. |
| `started` | boolean | Runtime ready flag. |
| `starting` | boolean | Startup in progress flag. |
| `mode` | string | Current mode, `final_signal_studio`. |
| `streamlit_required` | boolean | False for final Signal Studio mode. |

### `POST /api/config`

Request body is a JSON object containing any allowed keys from `CONFIG_KEYS`.

Example:

```json
{
  "SEARCH_TOOL_TYPE": "BochaAPI",
  "QUERY_ENGINE_MODEL_NAME": "deepseek-chat",
  "LANGSMITH_TRACING": "True"
}
```

Response:

| Field | Type | Description |
| --- | --- | --- |
| `success` | boolean | Update result. |
| `config` | object | Updated visible configuration. |
| `message` | string | Error message when update fails. |

Common configuration example:

```json
{
  "SEARCH_TOOL_TYPE": "BochaAPI",
  "QUERY_ENGINE_BASE_URL": "https://api.deepseek.com",
  "QUERY_ENGINE_MODEL_NAME": "deepseek-chat",
  "MEDIA_ENGINE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "MEDIA_ENGINE_MODEL_NAME": "qwen-plus",
  "REPORT_ENGINE_BASE_URL": "https://api.deepseek.com",
  "REPORT_ENGINE_MODEL_NAME": "deepseek-chat",
  "REPORT_OUTPUT_LANGUAGE": "en"
}
```

## Coordinator APIs

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/coordinator/latest` | Load newest Coordinator artifact plus metadata and feedback summary. |
| `POST` | `/api/coordinator/run` | Start an integrated analysis task. |
| `GET` | `/api/coordinator/task/{task_id}` | Poll Coordinator task status. |
| `GET` | `/api/coordinator/feedback` | List saved feedback records. |
| `POST` | `/api/coordinator/feedback` | Save review or revision feedback. |

### `POST /api/coordinator/run`

Request:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | string | Usually yes | Analysis topic. If omitted, server may reuse latest artifact query. |
| `feedback` | string | No | Revision or refinement instruction. |

Validation:

| Rule | Behavior |
| --- | --- |
| Request body must be an object | Returns `400`. |
| Query must be available from the request or latest artifact | Returns `400`. |
| Sensitive input filter blocks query/feedback | Returns `400` with `error_code=sensitive_input`. |

Response:

```json
{
  "success": true,
  "task": {
    "task_id": "coord_...",
    "query": "example topic",
    "has_feedback": false,
    "status": "queued",
    "progress": 0,
    "message": "Coordinator task queued"
  }
}
```

### `GET /api/coordinator/latest`

Success response:

| Field | Type | Description |
| --- | --- | --- |
| `success` | boolean | True when latest output exists. |
| `output` | object | Coordinator artifact. |
| `metadata.path` | string | Local artifact path. |
| `metadata.modified_at` | string | File modification timestamp. |
| `metadata.archive_count` | number | Count of timestamped archives. |
| `metadata.schema_version` | string | Artifact schema version. |
| `feedback` | object | Recent feedback records and summary. |
| `observability` | object | LangSmith configuration summary. |

When no artifact exists, the endpoint returns `404` with `success=false` and metadata.

Minimal successful shape:

```json
{
  "success": true,
  "output": {
    "schema_version": "1.0",
    "query": "Public reaction to a new AI policy",
    "synthesis": {
      "summary": "...",
      "top_insights": [],
      "key_tensions": [],
      "overall_confidence": 0.72
    },
    "source_data": {
      "query_agent": {
        "total_sources": 12,
        "stance_distribution": {}
      }
    }
  },
  "metadata": {
    "path": "AgentCoordinator/cache/coordinator_output_latest.json",
    "schema_version": "1.0"
  }
}
```

### `POST /api/coordinator/feedback`

Request:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `feedback` | string | Yes | Concrete feedback text. |
| `query` | string | No | Related query. |
| `target` | string | No | Review target, such as `Evidence grounding`. |
| `action` | string | No | `Review`, `Revise`, or `Rerun`. |
| `priority` | string | No | `Normal`, `High`, or `Critical`. |
| `thread_id` | string | No | Related Coordinator thread id. |

Saved records are appended to `AgentCoordinator/cache/frontend_feedback.jsonl`.

Example:

```json
{
  "query": "Public reaction to a new AI policy",
  "target": "Evidence grounding",
  "action": "Revise",
  "priority": "High",
  "feedback": "Add more official-source evidence and explain the disagreement between industry and civil-society sources."
}
```

## Report APIs

All report endpoints are mounted under `/api/report`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/report/status` | ReportEngine readiness, input mode, current task. |
| `POST` | `/api/report/generate` | Start report generation. |
| `GET` | `/api/report/progress/{task_id}` | Get task progress. |
| `GET` | `/api/report/stream/{task_id}` | Server-Sent Events stream. |
| `GET` | `/api/report/result/{task_id}` | HTML report response. |
| `GET` | `/api/report/result/{task_id}/json` | JSON task plus HTML content. |
| `GET` | `/api/report/download/{task_id}` | HTML file download. |
| `POST` | `/api/report/cancel/{task_id}` | Cancel a report task. |
| `GET` | `/api/report/templates` | Available report templates. |
| `GET` | `/api/report/log` | Report log lines. |
| `POST` | `/api/report/log/clear` | Clear report log. |
| `GET` | `/api/report/export/md/{task_id}` | Export Markdown. |
| `GET` | `/api/report/export/pdf/{task_id}` | Export PDF. |
| `POST` | `/api/report/export/pdf-from-ir` | Export PDF from supplied Document IR. |

### `POST /api/report/generate`

Request:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | string | No | Report topic. Defaults to `Intelligent Sentiment Analysis Report`. |
| `custom_template` | string | No | Markdown report template override. |

Response:

| Field | Type | Description |
| --- | --- | --- |
| `success` | boolean | True if task started. |
| `task_id` | string | Report task id. |
| `task` | object | Initial task state. |
| `source` | string | `engine_files` or `coordinator_latest`. |
| `stream_url` | string | SSE endpoint path. |

Example response:

```json
{
  "success": true,
  "task_id": "report_20260701_120500_ab12cd",
  "source": "coordinator_latest",
  "stream_url": "/api/report/stream/report_20260701_120500_ab12cd",
  "task": {
    "status": "pending",
    "progress": 0
  }
}
```

### `GET /api/report/stream/{task_id}`

Content type:

```text
text/event-stream
```

Event names:

| Event | Description |
| --- | --- |
| `status` | Status update. |
| `stage` | Stage transition. |
| `progress` | Progress update. |
| `warning` | Non-fatal warning. |
| `html_ready` | HTML is ready to fetch. |
| `completed` | Task completed. |
| `error` | Terminal task event with error details. |
| `log` | Log message. |
| `heartbeat` | Keep-alive. |

Example event:

```text
id: 12
event: progress
data: {"type":"progress","task_id":"report_...","progress":45,"stage":"process_chapter"}
```

Clients should handle reconnects with `Last-Event-ID`; the backend replays cached task events where possible.

### Export Endpoints

| Endpoint | Response Type | Requirement |
| --- | --- | --- |
| `/api/report/download/{task_id}` | `text/html` attachment | Completed task with saved HTML file. |
| `/api/report/export/md/{task_id}` | `text/markdown` attachment | Completed task with saved IR. |
| `/api/report/export/pdf/{task_id}` | `application/pdf` | Completed task with saved IR and configured PDF runtime stack. |
| `/api/report/export/pdf-from-ir` | `application/pdf` | Request body contains `document_ir`. |

PDF export uses the WeasyPrint/Pango runtime stack documented in [Setup: PDF Export Dependencies](../operations/setup.md#3-pdf-export-dependencies).

## Legacy APIs

These endpoints remain available as compatibility surfaces around the primary Signal Studio path.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | Status of legacy tracked apps. |
| `GET` | `/api/start/{app_name}` | Start legacy app or forum. |
| `GET` | `/api/stop/{app_name}` | Stop legacy app or forum. |
| `GET` | `/api/output/{app_name}` | Read legacy app output. |
| `GET` | `/api/test_log/{app_name}` | Write test log message. |
| `POST` | `/api/search` | Forward search query to running legacy Streamlit apps. |
| `GET` | `/api/forum/start` | Start ForumEngine monitor. |
| `GET` | `/api/forum/stop` | Stop ForumEngine monitor. |
| `GET` | `/api/forum/log` | Read parsed ForumEngine log. |
| `POST` | `/api/forum/log/history` | Read ForumEngine log from byte position. |

## Socket.IO Events

| Event | Direction | Purpose |
| --- | --- | --- |
| `connect` | client -> server | Server emits `status`. |
| `request_status` | client -> server | Server emits `status_update` for legacy tracked apps. |
| `console_output` | server -> client | Legacy log updates. |

## Machine-Readable Contract

See [OpenAPI YAML](openapi.yaml) for a static contract covering the final Signal Studio, Coordinator, and ReportEngine REST APIs. Legacy Streamlit and Forum endpoints are documented above for compatibility but are not part of the final OpenAPI surface.
