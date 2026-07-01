# Troubleshooting

## Startup Diagnostics

| Symptom | Diagnostic Focus | Operator Action |
| --- | --- | --- |
| Signal Studio shell needs a fresh asset build | Frontend asset bundle version. | Run `cd frontend && npm run build`; restart Flask. |
| `/api/system/start` reports an initialization message | ReportEngine settings and startup log. | Check `REPORT_ENGINE_*` settings and `logs/report.log`. |
| Port 5000 already in use | Another Flask process is running. | Stop the old process or change `PORT`. |
| Vite dev server is not reaching APIs | Flask runtime and proxy target. | Start `python app.py` or the documented `uv run` backend command; verify `frontend/vite.config.js` proxy. |

## Analysis Diagnostics

| Symptom | Diagnostic Focus | Operator Action |
| --- | --- | --- |
| `Analysis query is required` | Topic input. | Enter a topic before running. |
| Sensitive input modal appears | Topic or feedback matched `config/sensitive_words.txt`. | Revise input or update filter configuration deliberately. |
| Coordinator task returns a provider/config message | LLM/search provider profile. | Check [Configuration](../reference/configuration.md). |
| Coordinator runs too long | Search/provider latency. | Monitor task progress; adjust provider selection or timeouts. |
| Latest output is ready for refresh | Coordinator artifact timestamp. | Click Refresh or call `/api/coordinator/latest`. |

## Report Diagnostics

| Symptom | Diagnostic Focus | Operator Action |
| --- | --- | --- |
| `/api/report/generate` asks for analysis input | Coordinator artifact source. | Complete an analysis run first or provide the report input path. |
| Report stream closes early | Browser/network SSE interruption. | Reload task status or rerun generation. |
| Markdown export needs a fresh task artifact | Saved IR location. | Regenerate report and check ReportEngine logs. |
| PDF export reports dependency guidance | WeasyPrint/Pango runtime path. | Use the Docker runtime path or follow [Setup: PDF Export Dependencies](setup.md#3-pdf-export-dependencies). |
| Charts need repair inspection | Widget JSON and chart repair log. | Check JSON diagnostic logs and regenerate with the evaluated provider profile. |

## Observability Diagnostics

| Symptom | Diagnostic Focus | Operator Action |
| --- | --- | --- |
| Monitor shows local-only trace | LangSmith tracing profile. | Set `LANGSMITH_TRACING=True` and `LANGSMITH_API_KEY` for remote trace retrieval. |
| Trace endpoint reports remote retrieval status | Endpoint, key, project, and network. | Check `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, and provider connectivity. |
| Monitor has local trace data | Coordinator artifact trace. | Use local replay for the acceptance review path. |

## Frontend Diagnostics

| Symptom | Diagnostic Focus | Operator Action |
| --- | --- | --- |
| Export buttons are inactive | Current report task state. | Generate report first. |
| Latest analysis view needs refresh | UI polling and artifact timestamp. | Click Refresh; check `/api/coordinator/latest`. |
| Settings save returns validation details | Allowlisted config body. | Use fields in the Settings drawer; check `/api/config` response. |

## Diagnostic Files

| File | Use |
| --- | --- |
| `logs/report.log` | Report generation stages and errors. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Latest analysis artifact. |
| `AgentCoordinator/cache/frontend_feedback.jsonl` | Saved feedback records. |
| `output/json_error_logs/` | Report JSON parsing/repair diagnostics. |
| `api_evaluation/results_*/manual_review.csv` | Provider evaluation review details. |

## When To Rerun Tests

| Change | Tests |
| --- | --- |
| Sensitive input behavior | `python -m unittest tests.test_sensitive_input_filter` |
| Coordinator to ReportEngine bridge | `python -m unittest tests.test_coordinator_report_bridge` |
| Report IR/table/engine quote sanitization | `python -m unittest tests.test_report_engine_sanitization` |
| Forum log parsing | `pytest tests/test_monitor.py -v` or `python tests/run_tests.py` |

Use the `uv run --python 3.11 --with-requirements requirements.txt ...` equivalents in [Testing](../quality/testing.md) when Python is installed through `uv` or not exposed on `PATH`.
