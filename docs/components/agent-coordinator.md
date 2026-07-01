# AgentCoordinator

AgentCoordinator is the integrated reasoning layer. It runs QueryEngine and MediaEngine, bridges their outputs, computes divergence, deliberates over tensions, fills gaps, separates facts from opinions, synthesizes the final analysis, and exports the Coordinator artifact.

![Coordinator graph](../assets/diagrams/exported/coordinator-graph.png)

Read the graph as fan-out, normalization, reasoning, synthesis, and report handoff. The full-size image is easier to inspect for node names: [`docs/assets/diagrams/exported/coordinator-graph.png`](../assets/diagrams/exported/coordinator-graph.png).

## Implementation

| Path | Purpose |
| --- | --- |
| `AgentCoordinator/coordinator.py` | Public `AgentCoordinator` class, async/sync run methods, artifact export. |
| `AgentCoordinator/graph/builder.py` | LangGraph topology. |
| `AgentCoordinator/graph/state.py` | Coordinator state contract. |
| `AgentCoordinator/graph/nodes/` | Graph nodes for agent runs, data bridge, divergence, deliberation, gap filling, synthesis. |
| `AgentCoordinator/coordinator_output_schema.py` | Stable `coordinator_output.json` builder. |
| `AgentCoordinator/utils/report_bridge.py` | Adapter for ReportEngine input. |
| `AgentCoordinator/cache/` | Runtime artifacts and feedback log. |

## Graph Topology

| Phase | Nodes | Output |
| --- | --- | --- |
| Fan-out | `query_agent`, `media_agent` | Independent evidence and media outputs. |
| Bridge | `data_bridge` | Normalized propositions and common synthesis context. |
| Divergence | `divergence_compute` | Pairwise divergence values and hotspots. |
| Deliberation | `perspective_gen`, `deliberation` | Perspectives, consensus, dissent. |
| Gap filling | `gap_detector_router`, `targeted_search` | Supplementary evidence loop when needed. |
| Bias/fact processing | `echo_chamber`, `fact_opinion` | Bias warnings, verified facts, opinions, frameworks. |
| Platform and synthesis | `platform_interpret`, `synthesis` | Platform readings and final insight set. |
| Report bridge | `report_agent` | Report-oriented output and final trace. |

## Public Methods

| Method | Purpose |
| --- | --- |
| `run(query, thread_id=None, progress_callback=None)` | Async full pipeline. Streams node updates when a callback is provided. |
| `run_sync(query, thread_id=None, progress_callback=None)` | Synchronous wrapper used by Flask background tasks. |
| `save_result(result, filename=None)` | Saves result output when invoked directly. |

## Coordinator Output

`_export_coordinator_output()` writes:

| File | Purpose |
| --- | --- |
| `AgentCoordinator/cache/coordinator_output_<YYYYMMDD_HHMMSS>.json` | Timestamped archive. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Stable latest artifact consumed by Signal Studio and ReportEngine. |

The artifact is built by `build_coordinator_output()`. See [Coordinator Output Schema](../reference/coordinator-output-schema.md).

## Progress Reporting

When Flask starts a run, it passes a progress callback to `AgentCoordinator.run_sync()`. The callback maps graph node names to UI progress and updates task fields.

| Node | UI Meaning |
| --- | --- |
| `query_agent` | Evidence retrieval and stance analysis. |
| `media_agent` | Media research. |
| `data_bridge` | Output normalization. |
| `divergence_compute` | Source disagreement scoring. |
| `deliberation` | Multi-perspective reasoning. |
| `targeted_search` | Supplementary retrieval. |
| `synthesis` | Final insight construction. |
| `report_agent` | Report-ready output packaging. |

## Runtime Diagnostics

| Condition | Behavior |
| --- | --- |
| QueryEngine diagnostic event | Diagnostic detail is captured in agent state and surfaced through `agent_errors`. |
| MediaEngine configured output | Coordinator includes live media output or matching cached media output. |
| Deliberation loop reaches its configured round budget | Router proceeds to bias/fact processing with the accumulated reasoning state. |
| Artifact export diagnostic event | Flask task reports structured status; latest timestamped artifacts remain inspectable. |

## Related Documents

- [QueryEngine](query-engine.md)
- [MediaEngine](media-engine.md)
- [Coordinator Output Schema](../reference/coordinator-output-schema.md)
- [Runtime Flow](../architecture/runtime-flow.md)
