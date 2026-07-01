# Security And Safety

The system is designed as an operator-focused analysis application with explicit safety, secret-handling, and shared-deployment controls.

## Sensitive Input Filter

| Mechanism | Description |
| --- | --- |
| Utility | `utils/sensitive_input_filter.py` |
| Config flag | `ENABLE_SENSITIVE_INPUT_FILTER` |
| Word list | `config/sensitive_words.txt` |
| Error code | `sensitive_input` |
| Frontend behavior | Signal Studio shows a blocking modal. |

Protected routes:

| Route | Checked Fields |
| --- | --- |
| `POST /api/coordinator/run` | `query`, `feedback` |
| `POST /api/report/generate` | `query`, `custom_template` |
| `POST /api/search` | `query` |

## Secrets

| Secret Type | Examples |
| --- | --- |
| LLM keys | `QUERY_ENGINE_API_KEY`, `MEDIA_ENGINE_API_KEY`, `REPORT_ENGINE_API_KEY` |
| Search keys | `TAVILY_API_KEY`, `BOCHA_WEB_SEARCH_API_KEY`, `ANSPIRE_API_KEY` |
| Observability | `LANGSMITH_API_KEY` |
| Database | `DB_USER`, `DB_PASSWORD` |

Guidelines:

| Rule | Reason |
| --- | --- |
| Do not commit real `.env` values. | Prevent credential exposure. |
| Prefer environment injection in deployment. | Avoid baking secrets into images. |
| Do not expose `/api/config` publicly without authentication. | It can reveal/edit sensitive runtime settings. |

## API Exposure Controls

| Area | Control | Shared-Deployment Extension |
| --- | --- | --- |
| Authentication | Operator deployment keeps runtime access trusted. | Add login/session or reverse-proxy authentication. |
| Config write endpoint | Allowlisted keys only. | Restrict to trusted operator roles. |
| Long-running tasks | Task APIs expose status and progress. | Add rate limits and quotas. |
| File exports | Task ids identify export targets. | Add authenticated access checks. |
| Shutdown endpoint | Runtime control endpoint is documented. | Restrict to an admin role. |

## LLM Safety

| Concern | Current Handling |
| --- | --- |
| Prompt/output language | English report rules in prompts and ReportEngine configuration. |
| Malformed structured output | Parser, repair, validation, and error logs. |
| Source grounding | QueryEngine trust scoring and evidence tables. |
| Provider selection | Stored benchmark and manual review. |

## Data Handling

| Data | Location | Handling Note |
| --- | --- | --- |
| Coordinator artifacts | `AgentCoordinator/cache/` | May contain analysis text and source URLs. |
| Feedback | `frontend_feedback.jsonl` | May contain operator comments. |
| Logs | `logs/` | May include provider errors and generated text. |
| Model datasets | `SentimentAnalysisModel/` | Treat as project data assets. |

## Shared Deployment Controls

| Area | Action |
| --- | --- |
| Authentication | Add login/session or reverse-proxy auth. |
| Authorization | Restrict config, shutdown, exports, and logs. |
| Secret management | Use platform secret store. |
| Rate limiting | Limit analysis and report generation starts. |
| Audit logging | Track operator actions and artifact exports. |
| Data retention | Define cleanup policy for cache, logs, output. |
| Network | Serve behind TLS and restrict CORS if exposed. |
