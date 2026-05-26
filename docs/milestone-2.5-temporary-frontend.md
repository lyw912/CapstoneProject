# 5.3. Temporary Frontend and Integration Progress

> Owner: Miao (mmy0302)
> Scope: Unified control panel, Streamlit sub-app embedding, real-time log console, LLM configuration panel, ForumEngine integration, ReportEngine bridge
> Status: Fully functional temporary frontend — all core workflows operational

---

## 5.3.1. Overview

The current frontend is a **temporary but fully functional integration layer** built with Flask + vanilla HTML/CSS/JS + Socket.IO. It serves three purposes:

1. **Unified control panel** — a single-page dashboard that hosts all subsystems under one URL
2. **Runtime configuration** — bidirectional sync with `.env` so API keys and model settings can be changed without restarting
3. **Real-time observability** — live console streaming from all subprocesses via WebSocket

> **"Temporary" means the UI styling is minimal (brutalist black-and-white), not that functionality is missing.** Every control flow — starting the system, running searches, viewing agent outputs, generating reports, and downloading results — is wired and working end-to-end.

---

## 5.3.2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (http://localhost:5000)                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  index.html  (Flask-rendered control panel)       │  │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐ ┌──────────┐    │  │
│  │  │ Search  │ │ Config  │ │ App  │ │ Console  │    │  │
│  │  │ Bar +   │ │ Modal   │ │ Tabs │ │ Output   │    │  │
│  │  │ Upload  │ │ (.env)  │ │      │ │ (live)   │    │  │
│  │  └─────────┘ └─────────┘ └──────┘ └──────────┘    │  │
│  │  ┌────────────────────────────────────────────┐   │  │
│  │  │  Embedded Content Area (iframes)           │   │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │  │
│  │  │  │ Media    │ │ Query    │ │ Forum /    │  │   │  │
│  │  │  │ Agent    │ │ Agent    │ │ Report     │  │   │  │
│  │  │  │ :8502    │ │ :8503    │ │ Preview    │  │   │  │
│  │  │  └──────────┘ └──────────┘ └────────────┘  │   │  │
│  │  └────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│  Socket.IO ── real-time log streaming                   │
└─────────────────────────────────────────────────────────┘
                          │
           Flask app.py (port 5000)
           ├── Streamlit subprocess: Media Agent (:8502)
           ├── Streamlit subprocess: Query Agent (:8503)
           ├── ForumEngine background thread
           └── ReportEngine blueprint (/api/report/*)
```

---

## 5.3.3. Feature Walkthrough

### 5.3.3.1. Main Control Panel (Flask index.html)

**URL:** `http://localhost:5000`

The landing page is a single-page application with the following zones:

| Zone | Description |
|---|---|
| **Search Bar** | Text input + Start button. Triggers Media Agent / Query Agent with the entered keyword. Supports template upload (`.md`, `.txt`) for custom report structure. |
| **LLM Configuration** | Modal dialog that reads and writes directly to the `.env` file. All API keys, base URLs, and model names can be changed while the system is stopped. Changes persist to disk immediately on save. |
| **App Switcher Tabs** | Four tabs — Media Agent, Query Agent, Forum, Report — switch between embedded views and console logs. Status indicator dots show running (green) / stopped (red) for each subsystem. |
| **Embedded Content** | `<iframe>` embeds for Streamlit sub-apps (Media :8502, Query :8503), plus native HTML containers for Forum chat and Report preview. |
| **Console Output** | Real-time log panel showing `stdout`/`stderr` from all subprocesses, streamed via Socket.IO. Each log line is tagged with the source app. |
| **System Controls** | "Save & Start System" in config modal starts all subsystems. "Shutdown System" button gracefully terminates all child processes. |
| **Status Bar** | Bottom bar showing WebSocket connection status and system time. |

*(Screenshot: Main control panel with all four tabs and console output visible)*

### 5.3.3.2. LLM Configuration Panel

**Access:** Click "LLM Configuration" button on the main page.

- Reads current values from the backend `.env` file via `GET /api/config`
- Displays all LLM settings in categorized fields (API keys, base URLs, model names for each agent)
- Password-style masking for API key fields with a toggle eye icon
- "Save" writes back to `.env` via `POST /api/config`
- "Save & Start System" saves config and launches all child processes
- The config modal is locked (auto-saves disabled) while the system is running to prevent hot-reload issues

*(Screenshot: Configuration modal with API key fields populated)*

### 5.3.3.3. Embedded Streamlit Apps

Two Streamlit applications run as subprocesses managed by Flask:

**Media Agent** (`:8502` — `media_engine_streamlit_app.py`)
- Multimodal content understanding (video, image, structured cards)
- Broad crawling across TikTok, Kuaishou, Xiaohongshu
- Search via Bocha API or Anspire API (configurable)
- Output: analysis report with charts embedded in the Streamlit UI

**Query Agent** (`:8503` — `query_engine_streamlit_app.py`)
- Web search powered by Tavily API
- DeepSeek-powered reasoning for public opinion analysis
- Output: structured analysis report with source citations

Both apps are embedded in the main panel as `<iframe>` elements and can be switched via the app tabs. Clicking a tab loads the corresponding iframe and switches the console log source to that app's output.

*(Screenshot: Media Agent Streamlit app embedded in main panel)*

*(Screenshot: Query Agent Streamlit app embedded in main panel)*

### 5.3.3.4. ForumEngine Integration

The ForumEngine is an intelligent multi-agent discussion forum that runs in the background:

- **Forum Chat Area** — native HTML container in the main panel showing agent-to-agent conversations
- **Forum Log** — `logs/forum.log` is tailed and streamed to the frontend via Socket.IO
- **Participants** — The Host (coordinator), Query Agent, and Media Agent exchange analysis results in a structured conversation format
- Messages are parsed from log lines (timestamp, source, content) and rendered as chat bubbles

*(Screenshot: Forum chat with agent conversation visible)*

### 5.3.3.5. ReportEngine Bridge

After both analysis agents complete their work:

- The **Report tab** becomes unlocked
- "Generate Final Report" button triggers `POST /api/report/generate`
- Status messages show generation progress (chapter by chapter)
- Once complete, download buttons become active:
  - **Download HTML** — the full interactive report
  - **Download PDF** — server-side rendered PDF via WeasyPrint
  - **Download MD** — raw Markdown source
- Report preview renders the generated HTML in an iframe within the panel

*(Screenshot: Report preview with download buttons active)*

### 5.3.3.6. Real-Time Console & Logging

- Socket.IO connection established on page load (`socket.io.js` CDN)
- Every subprocess line is emitted as a `console_output` event with `{app, line}` payload
- Console output is maintained per app in an in-memory buffer (max 200 lines per app)
- Switching tabs refreshes the console to show the selected app's logs
- Log files are also written to disk: `logs/media.log`, `logs/query.log`, `logs/forum.log`

---

## 5.3.4. Integration Status Matrix

| Integration Point | Status | Detail |
|---|---|---|
| Flask orchestrator ↔ Streamlit subprocesses | Done | Subprocess management with health checks, port conflict resolution, graceful shutdown |
| Frontend ↔ Backend config sync | Done | Bidirectional `.env` read/write via REST API |
| Frontend ↔ Subprocess logs | Done | Socket.IO real-time streaming with per-app log buffers |
| Frontend ↔ ForumEngine | Done | Log parsing + chat rendering |
| Frontend ↔ ReportEngine | Done | Generate + download (HTML/PDF/MD) |
| Cross-origin iframe embedding | Done | Streamlit launched with `--server.enableCORS false` to allow embedding |
| System lifecycle | Done | Start all, shutdown all, per-app restart via API |

---

## 5.3.5. Known Limitations (Temporary Nature)

These are limitations of the *current temporary frontend*, not of the system's capabilities:

1. **No authentication / access control** — The config panel and system controls are open to anyone who can reach port 5000. This is acceptable for local development but needs a login layer before any deployment beyond localhost.

2. **Brutalist styling** — The UI uses a minimal black-and-white aesthetic with border-based layout. No design system, no responsive breakpoints for mobile. Functional but not polished.

3. **No SPA framework** — The frontend is vanilla HTML/CSS/JS (~4700 lines in a single file). No React/Vue component model. State is managed via global variables and DOM queries. This is fine for current scope but would benefit from a framework migration for long-term maintainability.

4. **iframe embedding fragility** — Streamlit apps are embedded via iframe, which means:
   - Streamlit's own UI chrome (sidebar, header) still appears inside the embedded view
   - Cross-origin restrictions require `--server.enableCORS false`
   - The iframe reloads when switching tabs, losing scroll position

5. **No progressive web app features** — No service worker, offline support, or push notifications.

6. **Single-file HTML** — The entire frontend lives in `templates/index.html` (~1600 lines of CSS + ~3100 lines of JS). This was deliberately kept monolithic for rapid iteration during development.

---

## 5.3.6. Why "Temporary"?

| Reason | Plan |
|---|---|
| Development velocity | A single-file vanilla frontend allowed rapid integration testing without framework overhead |
| API-first design | The Flask REST API + Socket.IO layer is the stable contract; the frontend is a consumer that can be replaced independently |
| Future migration path | The API surface (`/api/config`, `/api/system/start`, `/api/system/shutdown`, `/api/report/*`, Socket.IO events) is well-defined and can be consumed by a React/Vue SPA without backend changes |
| Immediate milestone needs | The current UI is sufficient for supervisor demos, integration testing, and capstone presentation |

---

## 5.3.7. How to Launch

```bash
# 1. Configure .env (at minimum: ANSPIRE_API_KEY or BOCHA_WEB_SEARCH_API_KEY)
cp .env.example .env
# Edit .env with your API keys

# 2. Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

# 3. Start the system
python app.py

# 4. Open browser
# Main panel:     http://localhost:5000
# Media Agent:    http://localhost:8502
# Query Agent:    http://localhost:8503
```

---

## 5.3.8. Screenshots

> **Capture guide:** With `python app.py` running, open `http://localhost:5000` in a browser. Each subsection below describes the exact state to capture.

### 5.3.8.1. Main Control Panel

Open `http://localhost:5000` after the system is fully started (all three status indicators green). This captures:
- Search bar at top with "Enter content to analyze..." placeholder
- "LLM Configuration" button and template upload icon
- Four app switcher tabs: Media Agent / Query Agent / Forum / Report
- Embedded content area showing the default Media Agent Streamlit iframe
- Console output panel at bottom with startup log lines
- Status bar showing "Connected" and current time

### 5.3.8.2. LLM Configuration Modal

Click "LLM Configuration" button to open the modal:
- Shows all API key fields (masked as passwords) with toggle eye icons
- Base URL and model name fields for each agent
- "Refresh", "Save", and "Save & Start System" buttons at bottom
- Modal overlay with "Bidirectional sync with .env file" title

### 5.3.8.3. Media Agent Embedded View

With the "Media Agent" tab active:
- Streamlit app embedded in the main panel's iframe
- Shows "Multimodal Agent" title and description
- Search input area within the Streamlit UI

### 5.3.8.4. Query Agent Embedded View

Click "Query Agent" tab:
- Streamlit app embedded in the main panel's iframe
- Shows "Query Agent" title and description
- Web search configuration area within the Streamlit UI

### 5.3.8.5. Forum Chat

Click "Forum" tab:
- Chat-style interface showing multi-agent conversation
- Messages from Host, Query Agent, and Media Agent with color-coded labels
- Console panel switches to forum log output

### 5.3.8.6. Report Generation

After running a complete analysis, click "Report" tab:
- Engine status block showing connection state
- "Generate Final Report" button (primary)
- Download buttons: HTML, PDF, MD (enabled after generation)
- Report preview iframe showing the rendered HTML report
- Task progress area above the preview

### 5.3.8.7. Console Output

With any tab active, scroll through the console panel at bottom:
- Color-coded log lines with timestamps
- Each line tagged with source: `[media]`, `[query]`, or `[forum]`
- Real-time streaming visible during active analysis

### 5.3.8.8. Full Workflow Sequence

Capture a series showing the end-to-end workflow:
1. Enter a search query and click "Start"
2. Media Agent producing results in the embedded view
3. Query Agent producing results in the embedded view
4. Forum tab showing agent discussion
5. Report tab with final generated report and download buttons active
