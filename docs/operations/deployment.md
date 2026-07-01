# Deployment

The repository includes a Dockerfile and docker-compose file for containerized deployment.

## Container Image

`Dockerfile` uses:

| Layer | Purpose |
| --- | --- |
| `python:3.11-slim` | Base runtime. |
| System packages | Build tools, browser/media dependencies, GTK/Pango/Cairo stack, ffmpeg. |
| `uv` | Python package installation. |
| `requirements.txt` | Backend dependencies. |
| Playwright Chromium | Browser automation/search dependencies. |
| Application source | Full repository copied into `/app`. |
| Default command | `python app.py`. |

Exposed ports:

| Port | Purpose |
| ---: | --- |
| 5000 | Flask/Signal Studio. |
| 8501 | Legacy Streamlit port. |
| 8502 | Legacy MediaEngine Streamlit port. |
| 8503 | Legacy QueryEngine Streamlit port. |

## Docker Compose

`docker-compose.yml` builds the local image from `Dockerfile` and defines:

| Service | Purpose |
| --- | --- |
| `public-opinion-system` | Application container. |
| `db` | PostgreSQL 15 service. |

Mounted volumes:

| Host Path | Container Path | Purpose |
| --- | --- | --- |
| `./logs` | `/app/logs` | Logs. |
| `./output` | `/app/output` | ReportEngine output and Document IR. |
| `./AgentCoordinator/cache` | `/app/AgentCoordinator/cache` | Coordinator artifacts and feedback log. |
| `./final_reports` | `/app/final_reports` | Report outputs. |
| `./.env` | `/app/.env` | Runtime configuration. |
| `./insight_engine_streamlit_reports` | `/app/insight_engine_streamlit_reports` | Legacy output. |
| `./media_engine_streamlit_reports` | `/app/media_engine_streamlit_reports` | Legacy output. |
| `./query_engine_streamlit_reports` | `/app/query_engine_streamlit_reports` | Legacy output. |

## Build Frontend Before Image

The container serves built frontend assets from `static/signal-studio/`. Build the frontend before building the image unless your deployment pipeline runs the frontend build separately.

```powershell
cd frontend
npm install
npm run build
cd ..
```

## Compose Run

The Compose file builds `public-opinion-system:latest` locally, so no registry image is required for a local deployment.

```powershell
docker compose up -d
```

Open:

```text
http://127.0.0.1:5000
```

If you want to push the image to a registry, tag and push `public-opinion-system:latest`, then replace the `build:` section in `docker-compose.yml` with the registry image name for that environment.

## Shared Deployment Controls

| Area | Control |
| --- | --- |
| Secrets | Do not bake real `.env` values into the image. Mount or inject secrets at runtime. |
| TLS | Put Flask behind a reverse proxy with TLS. |
| Persistence | Mount `AgentCoordinator/cache`, `output`, and `logs` if artifacts must survive container replacement. |
| Health checks | Add HTTP checks for `/api/system/status` or a dedicated health endpoint. |
| Task durability | For shared production use, replace in-memory task registries with durable storage. |
| Artifact retention | Add cleanup policy for `AgentCoordinator/cache/` and `output/`. |

## Related Documents

- [Setup](setup.md)
- [Configuration](../reference/configuration.md)
- [Runtime Assets](../reference/runtime-assets.md)
