# Setup

This guide starts the project from source on a local workstation.

## Prerequisites

| Tool | Purpose |
| --- | --- |
| Python 3.11+ | Backend runtime. Use system Python, Conda, or `uv` managed Python. |
| Node.js and npm | Build or run Signal Studio frontend. |
| Chromium dependencies for Playwright | Search/crawler-related dependencies. |
| WeasyPrint/Pango system libraries | PDF export stack. |
| API keys | LLM and search providers for live runs. |
| `uv` recommended | Runs Python commands with a managed Python environment. |

Verify the main tools before installing dependencies:

```powershell
python --version
uv run python --version
npm --version
docker --version
```

On Windows, install Python from python.org, winget, Conda, or use `uv` if `python --version` is not found. On macOS/Linux, use your system package manager, pyenv, Conda, or `uv`. Do not downgrade below Python 3.11.

## 1. Create Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

macOS/Linux shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

If using Conda, create an equivalent Python 3.11 environment and install the same requirements.

Conda:

```powershell
conda create -n capstone-project python=3.11
conda activate capstone-project
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

`uv` without a manually activated Python:

```powershell
uv run --python 3.11 --with-requirements requirements.txt python -m playwright install chromium
```

Then prefix Python commands with `uv run --python 3.11 --with-requirements requirements.txt` when you do not have an activated environment.

## 2. Configure `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum live-run settings:

| Area | Required Keys |
| --- | --- |
| QueryEngine LLM | `QUERY_ENGINE_API_KEY`, `QUERY_ENGINE_BASE_URL`, `QUERY_ENGINE_MODEL_NAME` |
| MediaEngine LLM | `MEDIA_ENGINE_API_KEY`, `MEDIA_ENGINE_BASE_URL`, `MEDIA_ENGINE_MODEL_NAME` |
| ReportEngine LLM | `REPORT_ENGINE_API_KEY`, `REPORT_ENGINE_BASE_URL`, `REPORT_ENGINE_MODEL_NAME` |
| Search | `SEARCH_TOOL_TYPE` plus the matching search key; default profile uses `TavilyAPI` with `TAVILY_API_KEY` |

Recommended provider choices are in [API Evaluation](../quality/api-evaluation.md).

For artifact-based review, use [Artifact Review](artifact-review.md).

## 3. PDF Export Dependencies

PDF export uses WeasyPrint and the Pango/Cairo stack. The Docker image includes this runtime path, and workstation installs can use the platform package route below.

| Platform | Recommended Path |
| --- | --- |
| Windows | Use the Docker image for the simplest PDF path, or install GTK/Pango runtime libraries compatible with WeasyPrint. |
| macOS | Install WeasyPrint system dependencies through Homebrew. |
| Debian/Ubuntu | Install Pango, Cairo, GDK-PixBuf, and related font/rendering libraries, or use the provided Dockerfile. |
| Docker | The Dockerfile installs the required GTK/Pango/Cairo libraries. |

## 4. Build Signal Studio

For production-style Flask serving:

```powershell
cd frontend
npm install
npm run build
cd ..
```

This writes assets to `static/signal-studio/`, which Flask serves through `templates/index.html`.

For frontend development:

```powershell
cd frontend
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:5000`.

## 5. Start Backend

```powershell
python app.py
```

Conda uses the same command after `conda activate capstone-project`.

`uv` equivalent:

```powershell
uv run --python 3.11 --with-requirements requirements.txt python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 6. Start Runtime In UI

In Signal Studio:

1. Open Settings and confirm provider keys.
2. Click **Save and Start Runtime**, or use the Monitor page runtime button.
3. Enter a topic in the brief input.
4. Click **Run**.
5. Generate a report from the Edit page when analysis completes.

## Verification Checklist

| Check | Expected |
| --- | --- |
| `GET /api/system/status` | JSON response with `mode=final_signal_studio`. |
| Signal Studio loads | JS/CSS assets render the application shell. |
| Settings load | `/api/config` returns visible keys. |
| Runtime starts | `/api/system/start` returns success and ReportEngine initialized. |
| Analysis starts | `/api/coordinator/run` returns a `coord_...` task. |
| Latest artifact exists | `/api/coordinator/latest` returns `output`. |
| Report generates | `/api/report/generate` returns `report_...` task and `stream_url`. |

## Related Documents

- [Runbook](runbook.md)
- [Artifact Review](artifact-review.md)
- [Configuration](../reference/configuration.md)
- [Troubleshooting](troubleshooting.md)
