# AgentCoordinator

AgentCoordinator remains the public analysis boundary. Existing callers still import `AgentCoordinator`, call `run()` or `run_sync()`, and read `AgentCoordinator/cache/coordinator_output_latest.json`. Internally, a parent LangGraph delegates typed tasks to the QueryEngine and MediaEngine subgraphs, reduces their contributions through one append-only Evidence Blackboard, and exports the long-standing Coordinator artifact fields as views over that evidence state.

## Active Runtime

| Phase | Runtime Node | Output |
| --- | --- | --- |
| Planning | `investigation_plan` | Budgeted Query and Media `ResearchTask` objects. |
| Specialist fan-out | `specialist_fanout` | Query breadth/stance contribution and Media section dossiers run in parallel. |
| Evidence reduction | `evidence_reduce` | Single-writer source canonicalization, acquisition observations, quality features, spans, and merged claim ledger. |
| Global loop | `global_sufficiency_audit` | Routes typed stance/official/counter-evidence tasks to Query and section-depth tasks to Media. |
| Audit and synthesis | `final_audit_synthesis` | Claim-level `AuditDecision` records and citation-bound insights. |

`AgentCoordinator.graph` now returns the active parent fusion graph. The earlier QueryAgent/DataBridge/deliberation/targeted-search graph remains in `AgentCoordinator/graph/` for historical comparison, but it is not the active endpoint and its lossy DataBridge is not reused.

## Implementation

| Path | Purpose |
| --- | --- |
| `AgentCoordinator/coordinator.py` | Stable public Coordinator class and artifact export. |
| `AgentCoordinator/fusion/` | Active parent LangGraph, typed fan-out/fan-in, follow-up routing, checkpoint namespace, and budgets. |
| `AgentCoordinator/intelligence/evidence_core/` | Single-writer blackboard, canonical source reducer, quality/claim merge, audit, and synthesis kernels. |
| `QueryEngine/contribution.py`, `MediaEngine/contribution.py` | Native specialist output projections. |
| `AgentCoordinator/intelligence/projection/report_engine_contract.py` | Projection from `CoordinatorIntelligenceArtifact` to the stable Coordinator JSON contract. |
| `AgentCoordinator/coordinator_output_schema.py` | Compatibility schema helper kept for tests and older callers; current endpoint export schema is `2.1-coordinator-intelligence`. |
| `AgentCoordinator/cache/` | Latest and timestamped runtime artifacts. Generated timestamped cache files are ignored. |

`AgentCoordinator/graph/` is no longer the active graph. The current `/api/coordinator/run` path uses `AgentCoordinator/fusion/graph.py`.

## Coordinator Output

`_export_coordinator_output()` writes:

| File | Purpose |
| --- | --- |
| `AgentCoordinator/cache/coordinator_output_<YYYYMMDD_HHMMSS>_<nanosecond-suffix>.json` | Collision-resistant runtime archive. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Stable latest artifact consumed by Signal Studio and ReportEngine. |

The artifact uses `schema_version = "2.1-coordinator-intelligence"` and contains `coordinator_intelligence` as the internal evidence ledger. Compatibility fields such as `synthesis`, `source_data`, `fact_opinion_separation`, `divergence_matrix`, and `deliberation` are evidence-derived views. The `artifact_derivation` field records that relationship so downstream adapters do not treat them as independent conclusions.

## Progress Reporting

Flask passes a progress callback to `AgentCoordinator.run_sync()`. The UI timeline maps the current endpoint implementation nodes:

| Node | UI Meaning |
| --- | --- |
| `investigation_plan` | Typed specialist plan and budgets. |
| `specialist_fanout` | Query/Media subgraph execution. |
| `evidence_reduce` | Blackboard ingest, canonicalization, quality, and claim ledger. |
| `global_sufficiency_audit` | Bounded follow-up decision. |
| `final_audit_synthesis` | Claim audit and final cited insight construction. |

## Runtime Diagnostics

| Condition | Behavior |
| --- | --- |
| Missing semantic provider | Deterministic quality rules remain active; server instrumentation should record the optional provider state. |
| Configured provider failure | Artifact records `provider:error` with the provider exception. |
| Specialist failure | The other specialist contribution remains usable; the failed run and diagnostic are preserved, and the artifact cannot imply that the failed specialist ran successfully. |
| Unsupported strong claim | Claim is weakened or rejected before synthesis. |

## Related Documents

- [QueryEngine](query-engine.md)
- [Configuration](../reference/configuration.md)
- [Runtime Flow](../architecture/runtime-flow.md)
