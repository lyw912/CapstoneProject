# ReportEngine

ReportEngine turns structured analysis into editable and exportable reports. It exposes the `/api/report/*` Blueprint, runs report tasks asynchronously, generates a Document IR, and renders HTML, Markdown, and PDF.

![ReportEngine pipeline](../assets/diagrams/exported/report-engine-pipeline.png)

Read this pipeline from template selection through chapter generation to Document IR and export. The graph is intentionally tall; use the full-size image for labels: [`docs/assets/diagrams/exported/report-engine-pipeline.png`](../assets/diagrams/exported/report-engine-pipeline.png).

## Implementation

| Path | Purpose |
| --- | --- |
| `ReportEngine/flask_interface.py` | Flask Blueprint, task registry, SSE stream, report endpoints. |
| `ReportEngine/agent.py` | `ReportAgent`, graph orchestration, input normalization, saving. |
| `ReportEngine/graph/builder.py` | LangGraph topology. |
| `ReportEngine/graph/nodes/` | Template, layout, chapter processing, finalize nodes. |
| `ReportEngine/ir/schema.py` | Document/chapter IR schema constants. |
| `ReportEngine/ir/validator.py` | IR validation. |
| `ReportEngine/renderers/` | HTML, Markdown, PDF, chart, math renderers. |
| `ReportEngine/report_template/` | Runtime report templates loaded by `TEMPLATE_DIR`. |
| `ReportEngine/utils/` | JSON parsing, chart validation/repair, dependency checks. |

## Graph Stages

| Stage | Responsibility |
| --- | --- |
| `template_selection` | Choose a report template or accept a custom template. |
| `template_slice` | Parse the template into sections/chapters. |
| `document_layout` | Design title, TOC, theme, and structure. |
| `word_budget` | Allocate word targets by chapter. |
| `prepare_storage` | Prepare manifest and chapter output storage. |
| `process_chapter` | Generate and repair chapter JSON; loops until all chapters are done. |
| `finalize_report` | Stitch chapters into Document IR, render HTML, and persist outputs. |

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /api/report/latest` | Newest completed report, including HTML and Document IR when available. |
| `GET /api/report/status` | ReportEngine readiness and current task state. |
| `POST /api/report/generate` | Start report generation. |
| `GET /api/report/progress/{task_id}` | Poll task state. |
| `GET /api/report/stream/{task_id}` | Server-Sent Events stream. |
| `GET /api/report/result/{task_id}` | HTML response for completed report. |
| `GET /api/report/result/{task_id}/json` | Task metadata plus HTML content. |
| `GET /api/report/download/{task_id}` | HTML file download. |
| `POST /api/report/cancel/{task_id}` | Cancel a pending/running task. |
| `GET /api/report/templates` | Available Markdown templates. |
| `GET /api/report/log` | Report log lines. |
| `POST /api/report/log/clear` | Clear report log. |
| `POST /api/report/render-ir` | Render supplied Document IR to HTML for the editor preview. |
| `GET /api/report/export/md/{task_id}` | Markdown export from Document IR. |
| `POST /api/report/export/md-from-ir` | Markdown export from supplied Document IR JSON. |
| `GET /api/report/export/pdf/{task_id}` | PDF export from Document IR. |
| `POST /api/report/export/pdf-from-ir` | PDF export from supplied Document IR JSON. |

Full API details: [API Reference](../reference/api.md).

## Input Modes

| Mode | Trigger | Description |
| --- | --- | --- |
| `engine_files` | Legacy Query/Media output files are ready. | Generates from legacy engine report files and forum logs. |
| `coordinator_latest` | Legacy files are not ready but latest Coordinator artifact exists. | Uses `AgentCoordinator/cache/coordinator_output_latest.json`. This is the final Signal Studio path. |

## Task Lifecycle

| Status | Meaning |
| --- | --- |
| `pending` | Task object created but not running yet. |
| `running` | Report generation is executing. |
| `completed` | HTML and associated files are ready. |
| `error` | Terminal diagnostic state with message. |
| `cancelled` | Task was cancelled through API. |

## SSE Event Types

| Event | Purpose |
| --- | --- |
| `status` | Task status or queue update. |
| `stage` | Major graph stage transition. |
| `progress` | Progress percentage and stage metadata. |
| `warning` | Non-blocking diagnostic event. |
| `html_ready` | HTML can be fetched. |
| `completed` | Terminal success event. |
| `error` | Terminal diagnostic event. |
| `log` | Log line forwarded from ReportEngine. |
| `heartbeat` | Keep-alive event. |

## Document IR

ReportEngine renders from an intermediate representation rather than directly from raw LLM prose. This allows validation, repair, and multiple output formats.

| IR Feature | Purpose |
| --- | --- |
| Chapter schema | Keeps generated chapters structured and ordered. |
| Block types | Supports rich report layout: tables, callouts, KPIs, widgets, figures, math. |
| Engine quotes | Preserves evidence from QueryAgent and MediaAgent with controlled titles. |
| Renderers | HTML, Markdown, and PDF use the same IR source. |
| Editor bridge | Signal Studio maps editable IR blocks into TipTap content and re-renders through `/api/report/render-ir`. |

See [Report IR](../reference/report-ir.md).

## Dependencies

| Dependency | Used For |
| --- | --- |
| `weasyprint` and system Pango stack | PDF export runtime stack. |
| Chart.js and related bundled libraries | Chart/widget rendering in HTML/PDF contexts. |
| MathJax/math renderer utilities | Math block rendering. |
| LLM provider settings | Template planning, chapter generation, repair. |

## Related Documents

- [Report IR](../reference/report-ir.md)
- [Runtime Assets](../reference/runtime-assets.md)
- [Troubleshooting](../operations/troubleshooting.md)
