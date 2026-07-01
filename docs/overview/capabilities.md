# Capabilities

This document summarizes what the system can do and where each capability is implemented.

## Capability Map

| Capability | User-Visible Result | Backend Contract | Main Modules |
| --- | --- | --- | --- |
| Runtime readiness | Signal Studio shows runtime status and can start ReportEngine. | `POST /api/system/start`, `GET /api/system/status` | `app.py`, `ReportEngine/flask_interface.py` |
| Integrated analysis | A topic becomes a structured analysis artifact. | `POST /api/coordinator/run`, `GET /api/coordinator/task/{task_id}` | `AgentCoordinator/` |
| Latest artifact loading | Readout, Proof, Monitor, and Edit pages receive the newest result. | `GET /api/coordinator/latest` | `app.py`, `AgentCoordinator/cache/` |
| Feedback loop | Revision requests are saved and can trigger a new analysis run. | `GET/POST /api/coordinator/feedback` | `app.py`, Signal Studio feedback drawer |
| Evidence review | Source table, stance mix, trust scores, and divergence heatmap. | Latest Coordinator artifact | `QueryEngine/`, `AgentCoordinator/`, Signal Studio Proof |
| Report editing | Operator can generate, edit, annotate, and export final narrative. | `POST /api/report/generate`, SSE stream, download/export endpoints | `ReportEngine/`, TipTap editor |
| Observability | Local trace replay and configurable LangSmith trace timeline. | `GET /api/observability/langsmith` | `app.py`, `langsmith` client, Signal Studio Monitor |
| Provider evaluation | Stored benchmark for LLM and search provider selection. | CLI harness, runtime configuration input | `api_evaluation/` |
| Regression verification | Tests for parsing, bridge contracts, sanitization, and filtering. | `pytest`, `unittest`, `tests/run_tests.py` | `tests/`, component utilities |

## Analysis Output

The integrated run produces `coordinator_output_latest.json` with these sections:

| Section | Purpose |
| --- | --- |
| `divergence_matrix` | Pairwise source-group disagreement and hotspots. |
| `deliberation` | Perspectives, phases, consensus, dissent, confidence. |
| `gap_filling` | Coverage gaps and supplementary retrieval rounds. |
| `platform_interpretations` | Platform-specific readings. |
| `bias_analysis` | Echo-chamber warnings and silent-majority hypothesis. |
| `fact_opinion_separation` | Verified facts, opinion/sentiment observations, analytical frameworks. |
| `synthesis` | Executive summary, insights, tensions, confidence, recommended follow-up. |
| `source_data` | QueryAgent source coverage, stance distribution, top sources, media availability. |
| `coordinator_trace` | Local execution trace for UI replay. |
| `agent_errors` | Diagnostic events recorded by agent nodes. |

See [Coordinator Output Schema](../reference/coordinator-output-schema.md) for the full contract.

## Report Output

ReportEngine converts analysis into a Document IR and renders:

| Output | Endpoint | Notes |
| --- | --- | --- |
| HTML preview | `GET /api/report/result/{task_id}` | Served as `text/html`. |
| HTML download | `GET /api/report/download/{task_id}` | Attachment download. |
| Markdown | `GET /api/report/export/md/{task_id}` | Generated from saved Document IR. |
| PDF | `GET /api/report/export/pdf/{task_id}` | Uses the configured WeasyPrint/Pango runtime stack. |

## Operational Envelope

| Area | Practical Handling |
| --- | --- |
| Provider quality | Use [Configuration](../reference/configuration.md) and [API Evaluation](../quality/api-evaluation.md) to select providers. |
| Legacy Streamlit endpoints remain in `app.py`. | Treat them as compatibility/operator utilities, not the primary final UI. |
| Some historical and runtime Markdown assets remain outside `docs/`. | They are documented in [Runtime Assets](../reference/runtime-assets.md). |
| Full end-to-end execution can be slow. | Use Monitor trace and Coordinator task polling; tune timeouts if needed. |
