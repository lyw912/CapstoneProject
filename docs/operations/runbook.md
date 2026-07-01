# Runbook

This runbook covers day-to-day local operation through Signal Studio and direct APIs.

## Standard Run

| Step | Action | Expected Result |
| --- | --- | --- |
| 1 | Start `python app.py`, or `uv run --python 3.11 --with-requirements requirements.txt python app.py` when Python is not on `PATH`. | Flask logs access URL. |
| 2 | Open `http://127.0.0.1:5000`. | Signal Studio loads. |
| 3 | Open Settings. | Config fields load from `/api/config`. |
| 4 | Save configuration and start runtime. | ReportEngine initializes; legacy Streamlit and Forum monitor remain stopped. |
| 5 | Enter topic and click Run. | Coordinator task appears. |
| 6 | Wait for completion. | Latest artifact updates. |
| 7 | Review Readout and Proof. | Synthesis, evidence, stance, and divergence are visible. |
| 8 | Generate report in Edit. | Report task streams events. |
| 9 | Export HTML/Markdown/PDF as needed. | Download endpoints return files. |

## Direct API Commands

Runtime status:

```powershell
curl.exe http://127.0.0.1:5000/api/system/status
```

Start runtime:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/system/start
```

Start analysis:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/coordinator/run `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Public reaction to a new AI policy\"}"
```

Load latest:

```powershell
curl.exe http://127.0.0.1:5000/api/coordinator/latest
```

Start report generation:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/report/generate `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Public reaction to a new AI policy\"}"
```

## Runtime State Locations

| Need | Path |
| --- | --- |
| Latest analysis | `AgentCoordinator/cache/coordinator_output_latest.json` |
| Archived analyses | `AgentCoordinator/cache/coordinator_output_*.json` |
| Feedback log | `AgentCoordinator/cache/frontend_feedback.jsonl` |
| Report log | `logs/report.log` |
| Report output | `output/` |
| Document IR | `output/document_ir/` |

## Operational Checks

| Check | How |
| --- | --- |
| ReportEngine ready | `GET /api/report/status` |
| Coordinator task status | `GET /api/coordinator/task/{task_id}` |
| Latest artifact metadata | `GET /api/coordinator/latest` |
| LangSmith trace status | `GET /api/observability/langsmith` |
| Report logs | `GET /api/report/log` |

## Stopping

Preferred UI path:

1. Use the shutdown button in Signal Studio.
2. Confirm the modal.

API path:

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/system/shutdown
```

## Legacy Operations

Legacy Streamlit and Forum controls are available but not required for final Signal Studio operation.

| Action | Endpoint |
| --- | --- |
| Start legacy app | `GET /api/start/{app_name}` |
| Stop legacy app | `GET /api/stop/{app_name}` |
| Read app output | `GET /api/output/{app_name}` |
| Start ForumEngine monitor | `GET /api/forum/start` |
| Stop ForumEngine monitor | `GET /api/forum/stop` |
| Read forum log | `GET /api/forum/log` |
