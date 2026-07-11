# Capabilities

This document summarizes what the system can do and where each capability is implemented.

## Capability Map

| Capability | User-Visible Result | Backend Contract | Main Modules |
| --- | --- | --- | --- |
| Runtime readiness | Signal Studio shows runtime status and can start ReportEngine. | `POST /api/system/start`, `GET /api/system/status` | `app.py`, `ReportEngine/flask_interface.py` |
| Integrated analysis | A topic becomes a structured analysis artifact. | `POST /api/coordinator/run`, `GET /api/coordinator/task/{task_id}` | `AgentCoordinator/` |
| Latest artifact loading | Readout, Proof, Monitor, and Edit pages receive the newest result. | `GET /api/coordinator/latest` | `app.py`, `AgentCoordinator/cache/` |
| Feedback loop | Revision requests are saved and can trigger a new analysis run. | `GET/POST /api/coordinator/feedback` | `app.py`, Signal Studio feedback drawer |
| Evidence review | Source table, stance mix, distinct evidence groups, repeated coverage, review outcomes, and provider diagnostics. | Latest Coordinator artifact | `AgentCoordinator/intelligence/`, Signal Studio Proof view |
| Report editing | Operator can generate, edit, annotate, and export final narrative. | `POST /api/report/generate`, SSE stream, download/export endpoints | `ReportEngine/`, TipTap editor |
| Observability | Local trace replay and configurable LangSmith trace timeline. | `GET /api/observability/langsmith` | `app.py`, `langsmith` client, Signal Studio Monitor |
| Provider evaluation | Stored benchmark for LLM and search provider selection. | CLI harness, runtime configuration input | `api_evaluation/` |
| Regression verification | Tests for parsing, bridge contracts, sanitization, and filtering. | `pytest`, `unittest`, `tests/run_tests.py` | `tests/`, component utilities |

## Analysis Output

The integrated run produces `coordinator_output_latest.json` with these sections:

| Section | Purpose |
| --- | --- |
| `coordinator_intelligence` | Internal EvidenceGraph-centered ledger with quality summaries, provider diagnostics, audit decisions, and cited insights. |
| `divergence_matrix` | Pairwise source-group disagreement and hotspots. |
| `deliberation` | Compatibility projection from claim-level audit and contradiction edges. |
| `gap_filling` | Retrieval tasks and follow-up rounds derived from adaptive research. |
| `platform_interpretations` | Platform-specific signal notes for observable social-platform samples; web-only runs use coverage context instead. |
| `bias_analysis` | Echo-chamber warnings and silent-majority hypothesis. |
| `fact_opinion_separation` | Verified facts, opinion/sentiment observations, analytical frameworks. |
| `synthesis` | Executive summary, insights, tensions, confidence, recommended follow-up. |
| `source_data` | Compatibility source summary derived from `coordinator_intelligence.evidence_graph`. |
| `coordinator_trace` | Local execution trace for UI replay. |
| `agent_errors` | Diagnostic events recorded by agent nodes. |

See [Coordinator Output Schema](../reference/coordinator-output-schema.md) for the full contract.

## Report Output

ReportEngine converts analysis into a Document IR and renders:

| Output | Endpoint | Notes |
| --- | --- | --- |
| HTML preview | `GET /api/report/result/{task_id}` | Served as `text/html`. |
| HTML download | `GET /api/report/download/{task_id}` | Attachment download. |
| Markdown | `GET /api/report/export/md/{task_id}` or `POST /api/report/export/md-from-ir` | Generated from saved or edited Document IR. |
| PDF | `GET /api/report/export/pdf/{task_id}` or `POST /api/report/export/pdf-from-ir` | Uses the configured WeasyPrint/Pango runtime stack. |

## Operational Envelope

| Area | Practical Handling |
| --- | --- |
| Provider quality | Use [Configuration](../reference/configuration.md) and [API Evaluation](../quality/api-evaluation.md) to select providers. |
| Legacy Streamlit endpoints remain in `app.py`. | Treat them as compatibility/operator utilities, not the primary final UI. |
| Some historical and runtime Markdown assets remain outside `docs/`. | They are documented in [Runtime Assets](../reference/runtime-assets.md). |
| Full end-to-end execution can be slow. | Use Monitor trace and Coordinator task polling; tune timeouts if needed. |
