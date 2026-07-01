# Contributing

This project is a capstone-oriented public-opinion intelligence system. Contributions should keep the final Signal Studio workflow, API contracts, and documentation aligned.

## Development Setup

Follow [Setup](docs/operations/setup.md). For artifact-based review, use [Artifact Review](docs/operations/artifact-review.md).

Recommended quick checks before opening a pull request:

```powershell
python -m unittest tests.test_sensitive_input_filter
python -m unittest tests.test_report_engine_sanitization
python -m unittest tests.test_coordinator_report_bridge
python -m unittest tests.test_media_agent_node_optional
pytest tests/test_monitor.py -v
```

If `python` is not available on `PATH`, use Conda or `uv` instead of skipping checks:

```powershell
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_sensitive_input_filter
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_report_engine_sanitization
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_coordinator_report_bridge
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_media_agent_node_optional
uv run --python 3.11 --with-requirements requirements.txt pytest tests/test_monitor.py -v
```

Conda equivalent: `conda activate capstone-project`, then run the standard commands.

## Change Guidelines

| Change Type | Required Updates |
| --- | --- |
| Public API change | Update `docs/reference/api.md` and `docs/reference/openapi.yaml`. |
| Coordinator artifact change | Update `docs/reference/coordinator-output-schema.md`, Signal Studio consumers, and bridge tests. |
| Report IR change | Update `docs/reference/report-ir.md`, validators, renderers, and sanitization tests. |
| Frontend workflow change | Update screenshots or `docs/components/signal-studio.md` if the visible workflow changes. |
| Provider/config change | Update `.env.example`, `docs/reference/configuration.md`, and `docs/quality/api-evaluation.md`. |
| Runtime asset movement | Update [Runtime Assets](docs/reference/runtime-assets.md) and [Data And Model Assets](docs/reference/data-and-model-assets.md). |

## Documentation Style

- Keep public docs in English.
- Prefer short explanation plus tables for scanability.
- Add examples for API or setup behavior that users may copy.
- Do not move runtime Markdown templates into `docs/`; some Markdown files are loaded by ReportEngine.
- Keep diagram source and exported PNG together under `docs/assets/diagrams/`.

## Security

- Do not commit real `.env` files, API keys, cookies, private prompts, or private feedback logs.
- Treat generated artifacts as potentially sensitive until reviewed.
- Do not expose `/api/config`, logs, exports, or shutdown endpoints publicly without authentication.

## Pull Request Checklist

| Check | Done |
| --- | --- |
| Tests or manual verification are described. |  |
| API/schema docs are updated when contracts change. |  |
| Setup/runbook docs are updated when commands change. |  |
| New data/model assets include source/license/privacy notes. |  |
| Screenshots or diagrams are updated when UI/architecture changes. |  |
