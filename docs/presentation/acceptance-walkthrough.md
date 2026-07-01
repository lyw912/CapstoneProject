# Acceptance Walkthrough

This walkthrough presents the final Signal Studio workflow for defense review and project acceptance.

## Pre-Walkthrough Checklist

| Check | Expected |
| --- | --- |
| `.env` configured | QueryEngine, MediaEngine, ReportEngine, and search provider profile are set. |
| Frontend built | `static/signal-studio/assets/app.js` and CSS exist. |
| Backend starts | `python app.py` or the documented `uv run` command starts Flask. |
| ReportEngine ready | `/api/system/start` succeeds from the UI or API. |
| Browser URL | `http://127.0.0.1:5000` opens Signal Studio. |

## Walkthrough Flow

| Step | Action | What To Point Out |
| --- | --- | --- |
| 1 | Open Signal Studio. | Single integrated workspace: Home, Readout, Proof, Edit, Monitor. |
| 2 | Open Settings. | Model/search providers, LangSmith tracing, runtime configuration. |
| 3 | Click Save and Start Runtime. | Final Signal Studio mode initializes ReportEngine and prepares runtime state. |
| 4 | Enter a topic. | The topic becomes the Coordinator input. |
| 5 | Click Run. | Coordinator task starts; Home shows task progress. |
| 6 | Wait for completion. | `coordinator_output_latest.json` is written and loaded by the UI. |
| 7 | Open Readout. | Synthesis, top insights, confidence, and tension view. |
| 8 | Open Proof. | Stance mix, divergence, source trust, platform interpretations. |
| 9 | Open Monitor. | Local trace replay, quality score, metadata, and LangSmith timeline when configured. |
| 10 | Open Edit and generate report. | ReportEngine starts; SSE stream shows graph stages. |
| 11 | Edit and annotate report. | HTML is editable in the review board. |
| 12 | Export report. | HTML, Markdown, and PDF endpoints are available after completion. |
| 13 | Save feedback. | Feedback is persisted and can launch refinement. |

## API Calls Behind The Walkthrough

| UI Action | API |
| --- | --- |
| Load status | `GET /api/system/status`, `GET /api/report/status`, `GET /api/config` |
| Start runtime | `POST /api/system/start` |
| Run analysis | `POST /api/coordinator/run` |
| Poll task | `GET /api/coordinator/task/{task_id}` |
| Load result | `GET /api/coordinator/latest` |
| Generate report | `POST /api/report/generate` |
| Stream report progress | `GET /api/report/stream/{task_id}` |
| Load report HTML | `GET /api/report/result/{task_id}/json` |
| Export report | `GET /api/report/download/{task_id}`, `GET /api/report/export/md/{task_id}`, `GET /api/report/export/pdf/{task_id}` |

## Artifact Review Points

| Evidence | Location | What It Proves |
| --- | --- | --- |
| Coordinator artifact | `AgentCoordinator/cache/coordinator_output_latest.json` | Analysis output is stable and reusable. |
| Media cache | `AgentCoordinator/cache/media_agent_<hash>.md` | MediaEngine output can be reused for matching topics. |
| Report outputs | `output/` and `static/v2_report_example/` | Generated reports have inspectable rendered artifacts. |
| OpenAPI contract | `docs/reference/openapi.yaml` | Runtime API surface is machine-readable. |
| Provider benchmark | `api_evaluation/results_full_r3/summary.csv` | Provider profile is evidence-based. |

## Closing Points

| Point | Evidence |
| --- | --- |
| The system is integrated | One UI controls analysis, evidence, reports, feedback, and monitoring. |
| The reasoning is inspectable | Proof page and Coordinator trace expose evidence and pipeline steps. |
| The output is reusable | Coordinator artifact feeds both UI and ReportEngine. |
| The report pipeline is structured | Document IR enables HTML, Markdown, and PDF export. |
| The provider choices are justified | API evaluation scores drive the recommended profile. |
