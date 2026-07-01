# System Architecture

CapstoneProject uses a layered architecture: Signal Studio provides the operator surface, Flask provides the runtime/API boundary, AgentCoordinator performs integrated reasoning, QueryEngine and MediaEngine collect and interpret evidence, and ReportEngine converts structured analysis into publication-ready output.

![System context](../assets/diagrams/exported/system-context.png)

Read this diagram from left to right: the operator starts in Signal Studio, Flask owns the runtime boundary, AgentCoordinator fans out to evidence engines, and ReportEngine consumes the stable Coordinator artifact. Open the full-size image at [`docs/assets/diagrams/exported/system-context.png`](../assets/diagrams/exported/system-context.png) if the Markdown preview is too small.

## Architectural Layers

| Layer | Responsibility | Primary Implementation |
| --- | --- | --- |
| Interface | Topic entry, analysis status, readout, evidence review, report editing, monitoring, configuration | `frontend/`, `templates/index.html`, `static/signal-studio/` |
| API gateway | HTTP APIs, background task management, config read/write, runtime state, static shell | `app.py` |
| Agent orchestration | Parallel engine runs, graph coordination, checkpointing, synthesis, artifact export | `AgentCoordinator/` |
| Evidence engines | Query planning, web/social retrieval, stance classification, media research | `QueryEngine/`, `MediaEngine/`, `MindSpider/` |
| Report generation | Template selection, chapter generation, Document IR, renderers, export APIs | `ReportEngine/` |
| Observability and quality | Local traces, LangSmith traces, tests, provider evaluation | `app.py`, `tests/`, `api_evaluation/` |

## Main Runtime Path

| Step | Action | System Boundary |
| --- | --- | --- |
| 1 | Operator enters a topic in Signal Studio. | Browser -> Flask |
| 2 | `POST /api/coordinator/run` creates a background Coordinator task. | Flask -> AgentCoordinator |
| 3 | AgentCoordinator runs QueryEngine and MediaEngine in parallel. | Coordinator -> engines |
| 4 | Coordinator computes divergence, deliberation, gap filling, bias checks, fact separation, and synthesis. | LangGraph pipeline |
| 5 | Coordinator writes `coordinator_output_latest.json`. | Local artifact |
| 6 | Signal Studio loads `/api/coordinator/latest`. | Flask -> browser |
| 7 | Operator starts report generation through `/api/report/generate`. | Browser -> ReportEngine Blueprint |
| 8 | ReportEngine streams progress through SSE and writes HTML/IR/state outputs. | ReportEngine -> browser/local disk |
| 9 | Operator edits, annotates, and exports HTML, Markdown, or PDF. | Browser -> ReportEngine export endpoints |

See [Runtime Flow](runtime-flow.md) for endpoint-level detail.

## Key Design Decisions

| Decision | Reason | Tradeoff |
| --- | --- | --- |
| Use Flask as the unified runtime gateway | Existing backend is Python-centric and already integrates ReportEngine, Coordinator, config, and static UI serving. | Long-running jobs require explicit background task and polling/SSE handling. |
| Use LangGraph for agent pipelines | Graph structure makes loops, fan-out/fan-in, and node-level progress easier to reason about. | State contracts must remain disciplined across nodes. |
| Persist a Coordinator artifact before report generation | Decouples analysis from report rendering and lets Signal Studio inspect the same structured output ReportEngine consumes. | Local artifact paths become part of runtime state. |
| Keep final Signal Studio mode separate from legacy Streamlit apps | Final UI is cohesive and does not use Streamlit sub-processes. | Compatibility endpoints remain documented as secondary surfaces. |
| Use Document IR for reports | Renderers can share a validated intermediate representation and support HTML/Markdown/PDF outputs. | LLM output must be repaired and validated before rendering. |

## Building Blocks

| Block | Inputs | Outputs | Operational Behavior |
| --- | --- | --- | --- |
| Signal Studio | Topic, settings, revision requests | API calls, edited report HTML, annotations | Shows API diagnostics and sensitive-input modal. |
| Flask Orchestrator | HTTP requests, config file, local artifacts | JSON responses, task state, Socket.IO events | Catches route exceptions and returns structured JSON diagnostics. |
| QueryEngine | Query string, provider settings | Structured source and stance output | Coverage loop and diagnostic log in state. |
| MediaEngine | Query string, provider settings | Media report text and search traces | Coordinator node uses configured providers and cached media output. |
| AgentCoordinator | Query, reviewer feedback | Coordinator output artifact | Captures `agent_errors`, exports trace and analysis context. |
| ReportEngine | Coordinator artifact or engine files, template | HTML, IR, Markdown, PDF | Task status, SSE diagnostics, JSON repair, validation, renderer recovery. |

## Cross-Cutting Concerns

| Concern | Mechanism |
| --- | --- |
| Configuration | `config.py` Pydantic Settings, `.env`, `/api/config` editing. |
| Secrets | API keys are configured through environment/config values and should not be committed. |
| Sensitive input | `utils/sensitive_input_filter.py` rejects blocked request text before analysis/report generation. |
| Traceability | Coordinator trace, local artifact metadata, configurable LangSmith tracing. |
| Output language | `REPORT_OUTPUT_LANGUAGE=en` and report prompts enforce English output by default. |
| Provider selection | `api_evaluation/` compares candidate LLM/search APIs before runtime adoption. |

## Related Documents

- [Runtime Flow](runtime-flow.md)
- [Data Artifacts](data-artifacts.md)
- [Quality Attributes](quality-attributes.md)
- [AgentCoordinator](../components/agent-coordinator.md)
- [ReportEngine](../components/report-engine.md)
