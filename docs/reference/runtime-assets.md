# Runtime Assets

Not every Markdown or static file is documentation. Some files are runtime assets, generated outputs, or model-adjacent notes. This document records the boundary.

## Documentation Assets

| Path | Purpose |
| --- | --- |
| `docs/` | Authoritative documentation library. |
| `docs/assets/diagrams/source/` | Excalidraw DSL and `.excalidraw` sources. |
| `docs/assets/diagrams/exported/` | Exported PNG diagrams used by Markdown pages. |
| `docs/assets/screenshots/` | Signal Studio screenshots retained as official visual assets. |

## Runtime Assets That Stay Outside `docs/`

| Path | Why It Stays |
| --- | --- |
| `ReportEngine/report_template/` | Loaded at runtime by `TEMPLATE_DIR`. |
| `ReportEngine/renderers/libs/` | Bundled JS libraries for report rendering. |
| `static/signal-studio/` | Generated frontend build output. |
| `templates/index.html` | Flask runtime HTML shell. |
| `static/v2_report_example/` | Example/generated report output. |
| `AgentCoordinator/cache/` | Runtime Coordinator artifacts and feedback log. |
| `output/`, `logs/` | Runtime-generated outputs and logs. |
| `SentimentAnalysisModel/**/README.md` | Model-family notes colocated with datasets/code. |

## Generated Presentation Artifacts

The repository may contain local PPT files and `ppt_work/` outputs. They are not required for the application runtime or the documentation library.

| Path Pattern | Status |
| --- | --- |
| `*.pptx` | Presentation artifacts. |
| `ppt_work/` | Presentation build scripts, screenshots, temporary extracted assets. |

## Historical Documentation

Older milestone documents and root summaries have been superseded by this documentation library.

| Previous Material | Replacement |
| --- | --- |
| Former root-level overview and progress notes | Root `README.md`, [Documentation Home](../README.md) |
| Former Query Agent summaries | [QueryEngine](../components/query-engine.md), [System Architecture](../architecture/system-architecture.md) |
| Former AgentCoordinator handoff/research notes | [AgentCoordinator](../components/agent-coordinator.md), [Coordinator Output Schema](coordinator-output-schema.md) |
| Former milestone reports | [Project Brief](../overview/project-brief.md), [Defense Brief](../presentation/defense-brief.md), [Assessment Matrix](../presentation/assessment-matrix.md) |
| Former test notes | [Testing](../quality/testing.md) |
| Former API evaluation notes | [API Evaluation](../quality/api-evaluation.md) plus source artifacts in `api_evaluation/` |

For dataset/model publication and clone-size guidance, see [Data And Model Assets](data-and-model-assets.md).

## Asset Maintenance Rules

| Rule | Reason |
| --- | --- |
| Do not move runtime templates unless `TEMPLATE_DIR` and tests are updated. | ReportEngine loads them by path. |
| Do not edit generated build assets directly. | Rebuild from `frontend/`. |
| Keep diagram source and export together. | Allows diagram edits without redrawing. |
| Keep generated artifacts out of documentation navigation unless they are intentional examples. | Avoids confusing source docs with runtime output. |
