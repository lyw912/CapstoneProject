# Multi-Agent Public Opinion Analysis System

This repository contains a Flask-based orchestration app that runs:

- `MediaEngine` (multimodal analysis)
- `QueryEngine` (web retrieval and stance-aware analysis)
- `ForumEngine` (agent discussion/coordination log monitor)
- `ReportEngine` (final report generation and export)

The frontend is served at `templates/index.html`, and backend orchestration lives in `app.py`.

## Runtime Architecture

At runtime:

1. `python app.py` starts only the Flask + Socket.IO server (default `:5000`).
2. From the web UI, `Save & Start System` triggers `/api/system/start`.
3. The backend starts:
   - `SingleEngineApp/media_engine_streamlit_app.py` on `:8502`
   - `SingleEngineApp/query_engine_streamlit_app.py` on `:8503`
   - Forum monitor thread
   - ReportEngine blueprint (already mounted under `/api/report`)
4. Search is sent to Media/Query Streamlit apps via iframe URL parameters.
5. ReportEngine reads latest outputs and generates HTML/PDF/Markdown.

Key references:

- `app.py`
- `ReportEngine/flask_interface.py`
- `SingleEngineApp/media_engine_streamlit_app.py`
- `SingleEngineApp/query_engine_streamlit_app.py`

## Requirements

- Python 3.10+ (3.11 recommended)
- Windows/Linux/macOS
- Chromium driver for Playwright:
  - `python -m playwright install chromium`
- Optional for PDF export:
  - WeasyPrint system dependencies (Pango/Cairo stack)

## Quick Start (Source Code)

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
cd D:\huang\Desktop\Capstone\CapstoneProject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
cd /path/to/CapstoneProject
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Prepare `.env`

```bash
cp .env.example .env
```

Fill at least these keys for full workflow:

- Media Agent:
  - `MEDIA_ENGINE_API_KEY`
  - `MEDIA_ENGINE_BASE_URL`
  - `MEDIA_ENGINE_MODEL_NAME`
- Query Agent:
  - `QUERY_ENGINE_API_KEY`
  - `QUERY_ENGINE_BASE_URL`
  - `QUERY_ENGINE_MODEL_NAME`
  - `TAVILY_API_KEY`
- Report Agent:
  - `REPORT_ENGINE_API_KEY`
  - `REPORT_ENGINE_BASE_URL`
  - `REPORT_ENGINE_MODEL_NAME`
- Forum Host:
  - `FORUM_HOST_API_KEY`
  - `FORUM_HOST_BASE_URL`
  - `FORUM_HOST_MODEL_NAME`
- Search backend (choose one):
  - `SEARCH_TOOL_TYPE=AnspireAPI` and `ANSPIRE_API_KEY`
  - or `SEARCH_TOOL_TYPE=BochaAPI` and `BOCHA_WEB_SEARCH_API_KEY`

Notes:

- `DB_*` is optional unless you explicitly use MindSpider/database workflows.
- If `.env` leaves base/model empty, provider calls will usually fail.

### 4. Start the system

```bash
python app.py
```

Open:

- Main UI: `http://localhost:5000`

Then in UI:

1. Open `LLM Configuration`
2. Save config
3. Click `Save & Start System`

## Running Components Individually (Optional)

Media Agent:

```bash
streamlit run SingleEngineApp/media_engine_streamlit_app.py --server.port 8502
```

Query Agent:

```bash
streamlit run SingleEngineApp/query_engine_streamlit_app.py --server.port 8503
```

## Main API Surface

### Orchestrator (`app.py`)

- `GET /api/status` - app process status
- `GET /api/start/<app_name>` - start one app (`media`, `query`, `forum`)
- `GET /api/stop/<app_name>` - stop one app
- `GET /api/output/<app_name>` - app log output
- `GET /api/config` - read config values
- `POST /api/config` - update config values
- `GET /api/system/status` - system lifecycle state
- `POST /api/system/start` - start all components
- `POST /api/system/shutdown` - graceful shutdown
- `GET /api/forum/log` - parsed forum log
- `POST /api/forum/log/history` - incremental forum log read

### ReportEngine (`/api/report/*`)

- `GET /api/report/status`
- `POST /api/report/generate`
- `GET /api/report/progress/<task_id>`
- `GET /api/report/stream/<task_id>` (SSE)
- `GET /api/report/result/<task_id>`
- `GET /api/report/result/<task_id>/json`
- `GET /api/report/download/<task_id>`
- `GET /api/report/export/md/<task_id>`
- `GET /api/report/export/pdf/<task_id>`
- `POST /api/report/export/pdf-from-ir`
- `POST /api/report/cancel/<task_id>`
- `GET /api/report/templates`

## Output and Logs

Common paths:

- `logs/`
  - `logs/media.log`
  - `logs/query.log`
  - `logs/forum.log`
  - `logs/report.log`
- Agent outputs:
  - `media_engine_streamlit_reports/`
  - `query_engine_streamlit_reports/`
- ReportEngine artifacts:
  - `output/`
  - `final_reports/` (if produced by current workflow)

## Testing

Examples:

```bash
python tests/run_tests.py
python -m unittest tests.test_coordinator_report_bridge
python -m unittest tests.test_report_engine_sanitization
```

## Docker (Current State)

`docker-compose.yml` exists, but the main service image is a placeholder:

- `your-registry/public-opinion-system:latest`

Replace it with a real image before using `docker compose up -d`.

## Troubleshooting

1. Port already in use (`5000`, `8502`, `8503`)
- Stop conflicting processes and restart.
- `app.py` tries to free occupied Streamlit ports, but not all cases are recoverable.

2. Report tab stays locked
- ReportEngine requires fresh Media + Query outputs and forum log input.
- Run a search first and wait until both agents complete.

3. Streamlit app loads but no analysis starts
- Check API keys in `.env`.
- Ensure search backend key matches `SEARCH_TOOL_TYPE`.

4. PDF export fails
- Usually missing WeasyPrint system dependencies.
- Use HTML/MD export first if environment is not ready for PDF.

