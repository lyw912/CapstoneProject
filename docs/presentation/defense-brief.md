# Defense Brief

This brief provides a concise technical narrative for presenting CapstoneProject.

## One-Minute Summary

CapstoneProject is a multi-agent public-opinion intelligence system. It starts from a topic, collects public evidence, scores and classifies sources, compares divergence across media and social signals, performs structured multi-perspective reasoning, and generates a traceable report through Signal Studio.

The key architectural decision is to split specialist research, evidence ownership, and document generation. QueryEngine owns breadth and stance coverage; MediaEngine owns narrative, section, and multimodal depth; AgentCoordinator owns typed delegation and global stopping; EvidenceCore owns canonical sources, acquisition provenance, quality, claims, relations, and audit; ReportEngine alone owns Document IR and rendering.

## Technical Claims

| Claim | Evidence In Repository |
| --- | --- |
| The system is modular rather than a monolithic prompt chain. | Active Query/Media subgraphs, parent fusion graph, shared Evidence Blackboard, EvidenceCore kernels, and unchanged ReportEngine boundary. |
| Multi-agent output cannot bypass evidence ownership. | `AcquisitionObservation`, `EvidenceSpan`, `ClaimProposal`, merged `Claim`, relation edges, and single-reducer ingest contracts. |
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
| QueryEngine | Active breadth specialist: stance planning, multi-source retrieval, trust/stance enrichment, coverage loop, and structured contribution. |
| MediaEngine | Active depth specialist: report structure, paragraph research, reflection, media framing, assets, and section dossiers. |
| AgentCoordinator | Parent supervisor: task budgets, parallel fan-out, typed follow-up routing, checkpoint namespace, stopping, and artifact projection. |
| EvidenceCore | Source truth owner: canonicalization, independent acquisition observations, quality, spans, claim merge, contradictions/support, and audit. |
| ReportEngine | Report-generation layer using templates, chapter JSON, Document IR, and renderers. |

## Diagrams To Use

| Diagram | Location | Best Use |
| --- | --- | --- |
| System context | `docs/architecture/system-architecture.md` Mermaid / `docs/assets/diagrams/source/system-context.dsl` | Current high-level architecture; regenerate exported PNG before use. |
| Final runtime | `docs/architecture/runtime-flow.md` Mermaid / `docs/assets/diagrams/source/final-runtime.dsl` | Current activate path; regenerate exported PNG before use. |
| Coordinator graph | `docs/components/agent-coordinator.md` | Active parent graph; do not present the historical exported Coordinator PNG as the current path. |
| QueryAgent graph | `docs/assets/diagrams/exported/query-agent-graph.png` | Evidence acquisition and stance coverage design reference. |
| ReportEngine pipeline | `docs/assets/diagrams/exported/report-engine-pipeline.png` | Structured report generation. |
| Artifact flow | `docs/assets/diagrams/exported/artifact-flow.png` | Analysis-to-report contract. |

Open full-size diagrams from `docs/assets/diagrams/exported/` during defense; several graphs are intentionally long and are easier to read outside the Markdown column.

## Current Strengths

| Strength | Why It Matters |
| --- | --- |
| Structured multi-agent pipeline | Query and Media both execute in the active path and submit inspectable typed batches. |
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
4. Explain [Query/Media Evidence Fusion](../architecture/query-media-evidence-fusion.md).
5. Explain [ReportEngine](../components/report-engine.md).
6. Explain [Contribution And Novelty](contribution-and-novelty.md).
7. Present Signal Studio using [Acceptance Walkthrough](acceptance-walkthrough.md).
8. Close with [Assessment Matrix](assessment-matrix.md) and [Evidence Dashboard](../quality/evidence-dashboard.md).

## Evidence Boundary

The fusion implementation and contract tests are repository evidence. They do not establish retrieval quality, factual accuracy, latency, cost, or superiority. Present the server demonstration and later experiments separately: Query-only, Media-only, previous intelligence path, and fused path on the same topic set, with coverage, citation validity, contradiction recall, runtime, provider calls, and failure cases.
