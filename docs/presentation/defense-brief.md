# Defense Brief

This brief provides a concise technical narrative for presenting CapstoneProject.

## One-Minute Summary

CapstoneProject is a multi-agent public-opinion intelligence system. It starts from a topic, collects public evidence, scores and classifies sources, compares divergence across media and social signals, performs structured multi-perspective reasoning, and generates a traceable report through Signal Studio.

The key architectural decision is to split the system into evidence engines, a reasoning coordinator, and a report engine. QueryEngine and MediaEngine collect and interpret evidence. AgentCoordinator performs divergence analysis, deliberation, gap filling, bias detection, and synthesis. ReportEngine converts the structured artifact into Document IR, then renders HTML, Markdown, and PDF.

## Technical Claims

| Claim | Evidence In Repository |
| --- | --- |
| The system is modular rather than a monolithic prompt chain. | Separate `QueryEngine/`, `MediaEngine/`, `AgentCoordinator/`, `ReportEngine/`, each with explicit graph/state contracts. |
| The final UI is an integrated workflow, not separate scripts. | Signal Studio in `frontend/`, served by Flask through `templates/index.html` and `static/signal-studio/`. |
| Analysis and reporting are decoupled by a stable artifact. | `AgentCoordinator/cache/coordinator_output_latest.json` and [Coordinator Output Schema](../reference/coordinator-output-schema.md). |
| The report pipeline is structured and validated. | `ReportEngine/ir/schema.py`, `ReportEngine/ir/validator.py`, renderer and sanitization tests. |
| Provider selection is evidence-driven. | `api_evaluation/` benchmark harness and results. |
| Safety and operability are first-class concerns. | Sensitive input filter, config drawer, runtime APIs, logs, LangSmith integration. |

See [Evidence Dashboard](../quality/evidence-dashboard.md) for the current benchmark and regression evidence, and [Contribution And Novelty](contribution-and-novelty.md) for the defense-oriented contribution map.

## Architecture Narrative

| Layer | Message |
| --- | --- |
| Signal Studio | A single operator surface for running analysis, inspecting evidence, editing reports, and monitoring trace quality. |
| Flask Orchestrator | The runtime boundary that serves UI, owns APIs, manages tasks, edits config, and initializes ReportEngine. |
| QueryEngine | Evidence and stance engine with search, deduplication, trust scoring, stance classification, and coverage loop. |
| MediaEngine | Media research engine for paragraph-level evidence synthesis. |
| AgentCoordinator | Reasoning layer that merges evidence, computes divergence, deliberates, fills gaps, separates facts/opinions, and synthesizes. |
| ReportEngine | Report-generation layer using templates, chapter JSON, Document IR, and renderers. |

## Diagrams To Use

| Diagram | Location | Best Use |
| --- | --- | --- |
| System context | `docs/assets/diagrams/exported/system-context.png` | High-level architecture. |
| Final runtime | `docs/assets/diagrams/exported/final-runtime.png` | End-to-end acceptance flow. |
| Coordinator graph | `docs/assets/diagrams/exported/coordinator-graph.png` | Core reasoning contribution. |
| QueryAgent graph | `docs/assets/diagrams/exported/query-agent-graph.png` | Evidence acquisition and stance coverage. |
| ReportEngine pipeline | `docs/assets/diagrams/exported/report-engine-pipeline.png` | Structured report generation. |
| Artifact flow | `docs/assets/diagrams/exported/artifact-flow.png` | Analysis-to-report contract. |

Open full-size diagrams from `docs/assets/diagrams/exported/` during defense; several graphs are intentionally long and are easier to read outside the Markdown column.

## Current Strengths

| Strength | Why It Matters |
| --- | --- |
| Structured multi-agent pipeline | Makes complex analysis inspectable and maintainable. |
| Evidence and reasoning separation | Avoids hiding evidence retrieval inside final report prose. |
| Stable Coordinator artifact | Enables UI, report generation, and debugging to consume the same data. |
| Document IR | Supports robust multi-format report rendering. |
| Provider evaluation harness | Reduces arbitrary model/API selection. |
| Final integrated UI | Makes the system reviewable as a workflow, not only backend modules. |

## Operational Envelope

| Area | Implemented Control | Extension Path |
| --- | --- |
| Provider quality and latency | `api_evaluation/` and runtime configuration select the provider profile. |
| Task continuity | Latest Coordinator and report artifacts persist to disk for review and handoff. |
| Compatibility endpoints | Documentation separates final Signal Studio APIs from compatibility surfaces. |
| UI acceptance | Screenshots and acceptance walkthrough cover the final operator workflow. |
| Artifact lifecycle | Runtime asset boundaries identify cache, output, and log paths. |

## Recommended Presentation Sequence

1. Start with [System Architecture](../architecture/system-architecture.md).
2. Show [Runtime Flow](../architecture/runtime-flow.md).
3. Explain [AgentCoordinator](../components/agent-coordinator.md).
4. Explain [ReportEngine](../components/report-engine.md).
5. Explain [Contribution And Novelty](contribution-and-novelty.md).
6. Present Signal Studio using [Acceptance Walkthrough](acceptance-walkthrough.md).
7. Close with [Assessment Matrix](assessment-matrix.md) and [Evidence Dashboard](../quality/evidence-dashboard.md).
