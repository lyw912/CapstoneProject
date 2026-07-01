# Testing

The repository contains focused regression tests for parsing, safety, report IR behavior, and integration contracts.

## Test Inventory

| Test File | Scope |
| --- | --- |
| `tests/test_monitor.py` | ForumEngine log parsing across old/new formats, JSON extraction, filtering, real examples. |
| `tests/test_sensitive_input_filter.py` | Sensitive input matching, fullwidth variants, payload shape. |
| `tests/test_report_engine_sanitization.py` | ReportEngine table repair, engine quote validation, sanitization rules. |
| `tests/test_coordinator_report_bridge.py` | Coordinator output adapter, ReportEngine bridge, English report directives. |
| `tests/test_media_agent_node_optional.py` | MediaAgent cache/config behavior in Coordinator integration. |
| `tests/run_tests.py` | Simple runner for ForumEngine parser tests. |
| `QueryEngine/evaluation/` | QueryAgent evaluation utilities. |
| `api_evaluation/` | Provider benchmark harness. |

## Recommended Commands

Run the focused unit/regression suite:

```powershell
python -m unittest tests.test_sensitive_input_filter
python -m unittest tests.test_report_engine_sanitization
python -m unittest tests.test_coordinator_report_bridge
python -m unittest tests.test_media_agent_node_optional
```

Run ForumEngine parser tests with pytest:

```powershell
pytest tests/test_monitor.py -v
```

If `python` is not available on `PATH`, run the same checks through `uv`:

```powershell
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_sensitive_input_filter
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_report_engine_sanitization
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_coordinator_report_bridge
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_media_agent_node_optional
uv run --python 3.11 --with-requirements requirements.txt pytest tests/test_monitor.py -v
```

Conda equivalent:

```powershell
conda activate capstone-project
python -m unittest tests.test_sensitive_input_filter
python -m unittest tests.test_report_engine_sanitization
python -m unittest tests.test_coordinator_report_bridge
python -m unittest tests.test_media_agent_node_optional
pytest tests/test_monitor.py -v
```

Direct runner:

```powershell
python tests/run_tests.py
```

Run API provider benchmark:

```powershell
cd api_evaluation
python run_evaluation.py --providers providers.local.json --out results_smoke_real
```

## Coverage Evidence

| Area | Coverage Evidence | Extension Target |
| --- | --- | --- |
| Sensitive input safety | Unit tests cover matching, disabled mode, fullwidth variants, response payload. | Route-level tests for `/api/coordinator/run` and `/api/report/generate`. |
| Report IR validation | Tests cover tables and engine quotes. | Full Document IR fixture tests for HTML/Markdown/PDF renderers. |
| Coordinator bridge | Tests cover adapter inputs and language directives. | Schema compatibility tests against `coordinator_output_latest.json` fixtures. |
| MediaAgent cache/config behavior | Tests cover configured output paths and Coordinator state integration. | Timeout and cache-hit scenarios. |
| Forum parser | Broad parser regression tests. | Compatibility coverage for new log formats. |
| Frontend | Acceptance walkthrough and screenshots document the main workflow. | Playwright flow for topic -> run -> latest -> report. |
| API reference | Static OpenAPI YAML exists. | Contract tests to ensure docs match routes. |

## Testing Strategy

| Change Type | Minimum Test Scope |
| --- | --- |
| API endpoint behavior | Add route tests and update [API Reference](../reference/api.md). |
| Coordinator output fields | Update schema doc, bridge tests, Signal Studio consumers. |
| Report IR block behavior | Add validator and renderer tests. |
| Frontend workflow | Add Playwright or component tests for affected view. |
| Provider configuration | Run `api_evaluation` smoke tests with configured providers. |
| Sensitive terms | Run sensitive input filter tests. |

## Manual Verification Checklist

| Scenario | Expected |
| --- | --- |
| Start runtime | `/api/system/start` succeeds and ReportEngine is initialized. |
| Run analysis | Coordinator task completes and latest artifact updates. |
| Review Proof page | Evidence table, stance chart, divergence heatmap render. |
| Generate report | SSE emits stages and completed event. |
| Export files | HTML, Markdown, and PDF endpoints return downloads. |
| Save feedback | Monitor shows saved request. |
| LangSmith local mode | Monitor shows local trace state. |

## Related Documents

- [API Evaluation](api-evaluation.md)
- [Security And Safety](security-and-safety.md)
- [Troubleshooting](../operations/troubleshooting.md)
