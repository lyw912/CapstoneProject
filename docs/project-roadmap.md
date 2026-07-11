# Roadmap

This roadmap records completed scope and the engineering evolution path for shared deployments.

## Completed Core Scope

| Area | Status | Evidence |
| --- | --- | --- |
| Final operator UI | Implemented | Signal Studio, screenshots, Flask-served build. |
| Integrated analysis path | Implemented | `/api/coordinator/run`, AgentCoordinator runtime and artifact export. |
| Evidence engine integration | Implemented | QueryEngine and MediaEngine under Coordinator. |
| Divergence and deliberation | Implemented | Coordinator graph nodes. |
| Report generation | Implemented | ReportEngine graph, Document IR, HTML/Markdown/PDF export. |
| Runtime API docs | Implemented | API Reference and OpenAPI YAML. |
| Operations docs | Implemented | Setup, artifact review, runbook, deployment, troubleshooting. |
| Provider benchmark | Implemented | `api_evaluation/` and Evidence Dashboard. |
| Media performance controls | Implemented | Parallel paragraph workers, request timeouts, stream-idle watchdog, Coordinator media cache. |

## Engineering Evolution

| Priority | Item | Reason |
| --- | --- | --- |
| High | Add route-level API tests | Keep Flask behavior and API docs aligned. |
| High | Add Playwright end-to-end tests | Cover topic launch, latest artifact, report generation, export, and feedback. |
| High | Add artifact retention job | Prevent unbounded cache/output/log growth. |
| Medium | Persist task registry | Preserve active task state across server restarts. |
| Medium | Add provider rotation policy | Improve live-run resilience when providers are slow. |
| Medium | Add OpenAPI validation in CI | Detect stale machine-readable contract. |

## Shared Deployment Controls

| Area | Target |
| --- | --- |
| Authentication | Protect configuration, logs, exports, shutdown, and analysis starts. |
| Authorization | Separate viewer, analyst, and admin permissions. |
| Rate limits | Prevent accidental provider quota exhaustion. |
| Secret management | Use platform secret stores instead of editable local config for shared deployments. |
| Durable storage | Replace in-memory task registries with database-backed state. |
| Observability | Add structured metrics for task durations, provider events, and export diagnostics. |

## Research Extensions

| Extension | Direction |
| --- | --- |
| Local sentiment adapters | Connect selected `SentimentAnalysisModel/` assets to QueryEngine social enrichment through a stable adapter. |
| Baseline evaluation | Compare single-prompt, basic RAG, and full Coordinator workflows on the same topics. |
| Report quality scoring | Add automated checks for evidence coverage, unsupported claims, and IR validity. |
| Multilingual operation | Expand controlled English/Chinese output handling beyond report generation. |
