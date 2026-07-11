# Project Brief

CapstoneProject is a multi-agent public-opinion intelligence system for turning an analysis brief into a traceable, evidence-grounded report. The final operator surface is Signal Studio, a React interface served by the Flask orchestrator.

## Core Outcome

Given a topic or issue, the system should:

1. Collect public evidence from search providers and social data sources.
2. Normalize evidence into source items, quality features, distinct source groups, and cited source excerpts.
3. Separate stance, sentiment, aspect, repeated coverage, freshness, and provider diagnostics.
4. Mine claims from representative evidence and route weak claims through adaptive follow-up search.
5. Audit claims before synthesis so unsupported strong statements are weakened or rejected.
6. Persist a structured Coordinator artifact with an internal evidence ledger and compatibility views.
7. Generate an editable report and export it as HTML, Markdown, or PDF.

## Primary Users

| User Type | Need | Supporting Surface |
| --- | --- | --- |
| Analyst | Run an investigation, inspect evidence, request revision | Signal Studio Home, Readout, Proof, Edit |
| Operator | Configure keys, start runtime, monitor trace quality | Signal Studio Monitor and Settings |
| Engineer | Maintain agents, API contracts, renderers, and deployment | `docs/components/`, `docs/reference/`, tests |
| Reviewer | Evaluate architecture, scope, acceptance evidence, and quality controls | `docs/presentation/`, `docs/quality/` |

## System Capabilities

| Capability | Description | Implemented By |
| --- | --- | --- |
| Integrated analysis run | Launches a full Coordinator pipeline from a topic. | `POST /api/coordinator/run`, `AgentCoordinator/` |
| Evidence acquisition | Generates retrieval tasks, calls configured search providers, and normalizes results. | `AgentCoordinator/intelligence/acquisition/` |
| Quality modeling | Builds distinct source groups, repeated-coverage counts, cited source excerpts, freshness, and quality summaries. | `AgentCoordinator/intelligence/quality/` |
| Adaptive research | Routes weak, one-sided, stale, or UGC-only claims through follow-up retrieval. | `AgentCoordinator/intelligence/reasoning/adaptive_loop.py` |
| Evidence audit | Produces claim-level accept/weaken/reject decisions before synthesis. | `AgentCoordinator/intelligence/reasoning/audit.py` |
| Citation synthesis | Generates final insights only from audited claims and cited sources. | `AgentCoordinator/intelligence/reasoning/synthesis.py` |
| Report generation | Converts structured analysis into Document IR and rendered output. | `ReportEngine/` |
| Monitoring | Shows local replay, quality metrics, LangSmith traces, feedback history. | Signal Studio Monitor, `/api/observability/langsmith` |

## Final Signal Studio Mode

The final runtime path is Signal Studio plus Flask APIs:

| Runtime Element | Final Runtime Behavior |
| --- | --- |
| Signal Studio | Primary UI. Served by Flask using `templates/index.html` and `static/signal-studio/`. |
| AgentCoordinator | Public analysis boundary. Calls `AgentCoordinator/intelligence/` internally and writes the Coordinator artifact. |
| ReportEngine | Initialized by `/api/system/start`; generates reports from Coordinator artifacts. |
| Legacy Streamlit apps | Explicitly stopped in `initialize_system_components()` for final Signal Studio mode. |
| Forum monitor | Managed as a compatibility surface while Signal Studio remains the primary path. |

## Operational Envelope

| Area | Impact |
| --- | --- |
| External APIs | LLM and search keys configure live analysis. |
| Search latency can dominate runtime | Source acquisition and semantic provider timeouts are explicit in configuration and diagnostics. |
| PDF export stack | WeasyPrint/Pango runtime path is documented for workstation and Docker use. |
| Generated artifacts | `AgentCoordinator/cache/` and `output/` are runtime state. |
| Sensitive input filter is enabled by default | Requests containing blocked terms are rejected before analysis/report generation. |

## Next Documents

- [Capabilities](capabilities.md)
- [System Architecture](../architecture/system-architecture.md)
- [API Reference](../reference/api.md)
- [Setup Guide](../operations/setup.md)
