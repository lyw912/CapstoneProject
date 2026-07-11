# Quality Attributes

This document records the main quality goals that shape the architecture.

## Quality Attribute Summary

| Attribute | Design Mechanism | Current Evidence |
| --- | --- | --- |
| Traceability | Coordinator trace, task progress, latest artifact metadata, LangSmith integration | Signal Studio Monitor, `/api/observability/langsmith` |
| Evidence grounding | Cited source excerpts, audited claims, distinct evidence groups, quality summaries, source coverage, provider diagnostics | Coordinator artifact and Signal Studio Proof view |
| Modularity | Coordinator boundary plus EvidenceGraph-centered internal layer and ReportEngine projection | `AgentCoordinator/`, `AgentCoordinator/intelligence/`, `ReportEngine/` |
| Report reliability | Document IR schema, validation, repair, renderer recovery | `ReportEngine/ir/`, sanitization tests |
| Operability | `/api/system/start`, `/api/system/shutdown`, config drawer, logs | Flask runtime APIs |
| Provider portability | OpenAI-compatible LLM settings and provider benchmark harness | `config.py`, `api_evaluation/` |
| Safety | Sensitive input filter, explicit secret settings, local config boundaries | `utils/sensitive_input_filter.py`, tests |

## Runtime Reliability

| Runtime Concern | Mitigation |
| --- | --- |
| Long-running analysis tasks block the UI | Coordinator and ReportEngine run in background threads; UI polls or listens via SSE. |
| Report SSE connection drops | Report stream replays historical events after `Last-Event-ID` and sends heartbeats. |
| Legacy process conflicts | Signal Studio startup stops legacy Streamlit and Forum monitor processes. |
| Provider diagnostics | Task status and artifact capture source acquisition, semantic provider, structured LLM, and replay diagnostics. |
| Malformed report JSON | ReportEngine uses validation, repair attempts, and diagnostic logs. |

## Maintainability

| Practice | Location |
| --- | --- |
| Runtime topology is explicit | `AgentCoordinator/intelligence/engine.py`, `AgentCoordinator/coordinator.py` |
| State contracts are typed | `AgentCoordinator/intelligence/contracts/`, `ReportEngine/ir/schema.py` |
| API surface is centralized | `app.py`, `ReportEngine/flask_interface.py` |
| Frontend API usage is encapsulated | `frontend/src/hooks/`, `frontend/src/utils/helpers.js` |
| Configuration is declared in one class | `config.py::Settings` |

## Known Tradeoffs

| Tradeoff | Explanation |
| --- | --- |
| Local artifacts instead of a durable job database | Simpler and transparent for the current system; less suitable for multi-user production. |
| Mixed final and legacy API surfaces | Keeps compatibility while the final UI stabilizes; requires clear documentation separation. |
| Provider flexibility increases configuration complexity | Enables evaluation and replacement but requires careful `.env` management. |
| LLM-generated structured output needs repair | Enables rich reports, but validation/repair adds runtime complexity. |

## Engineering Evolution

| Improvement | Reason |
| --- | --- |
| Add persistent task storage | Preserve job state across server restarts. |
| Add automated OpenAPI validation tests | Keep `docs/reference/openapi.yaml` aligned with route behavior. |
| Add link checking to CI | Prevent stale documentation links. |
| Add end-to-end Signal Studio tests | Cover the final UI workflow from topic to export. |
| Add artifact retention policy | Prevent unbounded cache/output growth. |
