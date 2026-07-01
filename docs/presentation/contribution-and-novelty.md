# Contribution And Novelty

This page summarizes the project contribution in a form suitable for defense review. It separates implemented work, evidence, novelty, and engineering evolution.

## Contribution Map

| Area | Contribution | Evidence |
| --- | --- | --- |
| Final product workflow | Built Signal Studio as a single operator surface for launch, readout, proof inspection, report editing, monitoring, and configuration. | `frontend/`, `static/signal-studio/`, [Screenshots](../overview/screenshots.md) |
| Runtime orchestration | Consolidated the final path behind Flask APIs with background Coordinator tasks, ReportEngine initialization, config editing, and observability endpoints. | `app.py`, [Flask Orchestrator](../components/flask-orchestrator.md) |
| Multi-agent reasoning | Integrated QueryEngine and MediaEngine under AgentCoordinator, then added divergence, deliberation, targeted search, fact/opinion separation, platform interpretation, and synthesis. | `AgentCoordinator/graph/`, [AgentCoordinator](../components/agent-coordinator.md) |
| Evidence pipeline | Implemented structured QueryEngine flow for query planning, search, deduplication, trust scoring, stance classification, coverage checks, and output assembly. | `QueryEngine/graph/`, [QueryEngine](../components/query-engine.md) |
| Report generation | Built ReportEngine around templates, chapter planning, Document IR, validation, repair, and HTML/Markdown/PDF renderers. | `ReportEngine/`, [ReportEngine](../components/report-engine.md), [Report IR](../reference/report-ir.md) |
| Traceability | Preserved a stable Coordinator artifact consumed by UI and ReportEngine, plus local traces and configurable LangSmith integration. | `AgentCoordinator/cache/`, [Data Artifacts](../architecture/data-artifacts.md) |
| Quality and safety | Added focused tests, sensitive input filtering, provider evaluation, security notes, setup/runbook/troubleshooting docs. | `tests/`, `api_evaluation/`, [Quality](../quality/testing.md) |

## What Is Different From Recognized Reference Points

| Reference Point | Known For | Project Difference |
| --- | --- | --- |
| [RAG](https://arxiv.org/abs/2005.11401) | Retrieval-grounded generation. | Adds stance, trust, divergence, and proof UI. |
| [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) | Graph reasoning over corpora. | Adds live public-opinion operation and exports. |
| [AutoGen](https://microsoft.github.io/autogen/stable/index.html) / [CrewAI](https://github.com/crewaiinc/crewai) | General multi-agent orchestration. | Adds domain agents and a stable Coordinator artifact. |
| Sentiment classifiers | Text labels and scores. | Adds investigation, synthesis, and reports. |
| BI/static dashboards | Visualization. | Adds run, refine, edit, monitor, export. |

## Novelty Claims

| Claim | Why It Matters | Evidence |
| --- | --- | --- |
| Artifact-centered agent workflow | Decouples live analysis, UI inspection, and report generation. | [Coordinator Output Schema](../reference/coordinator-output-schema.md) |
| Structured deliberation after retrieval | Treats public-opinion analysis as a reasoning problem, not only retrieval. | `AgentCoordinator/graph/nodes/deliberation_engine.py` |
| Divergence and gap handling | Surfaces disagreement and underrepresented perspectives before report writing. | `divergence_matrix_node.py`, `gap_detector.py`, `targeted_search_node.py` |
| Document IR for reports | Makes generated reports repairable, testable, and exportable across formats. | [Report IR](../reference/report-ir.md) |
| Operator-grade frontend | Combines evidence review, editing, monitoring, and configuration. | [Signal Studio](../components/signal-studio.md) |

## Workload Snapshot

| Workstream | Representative Files |
| --- | --- |
| Backend API and orchestration | `app.py`, `ReportEngine/flask_interface.py` |
| Agent graph implementation | `AgentCoordinator/graph/`, `QueryEngine/graph/`, `MediaEngine/graph/`, `ReportEngine/graph/` |
| Frontend application | `frontend/src/views/`, `frontend/src/components/`, `frontend/src/hooks/` |
| Report rendering and export | `ReportEngine/ir/`, `ReportEngine/renderers/`, `ReportEngine/utils/` |
| Evaluation and tests | `tests/`, `api_evaluation/`, `QueryEngine/evaluation/` |
| Documentation and operations | `README.md`, `docs/`, `Dockerfile`, `docker-compose.yml` |

## Engineering Evolution

| Area | Current Control | Extension Path |
| --- | --- | --- |
| Provider operations | Provider evaluation, OpenAI-compatible clients, timeout settings, and recommended profiles. | Add automatic provider rotation by engine. |
| Task continuity | Persisted Coordinator artifacts and generated report outputs. | Add database-backed active task registry. |
| UI acceptance | Screenshots, acceptance walkthrough, and frontend build artifacts. | Add Playwright end-to-end checks. |
| Shared deployment | Security model, secret handling, Docker deployment, and protected-route guidance. | Add authentication, authorization, and rate limits. |
| Asset lifecycle | Runtime asset boundaries and generated-output paths. | Add cleanup policy and scheduled retention job. |

## Defense Talking Points

1. The project contribution is not just model prompting; it is the integration of retrieval, reasoning, validation, UI, and operations.
2. The stable Coordinator artifact is the contract that keeps the system inspectable.
3. ReportEngine's Document IR is the reliability layer between LLM generation and exported documents.
4. Signal Studio presents the system as an operator workflow rather than disconnected scripts.
5. The engineering evolution items are shared-deployment controls around an already integrated workflow.
