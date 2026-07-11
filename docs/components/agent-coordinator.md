# AgentCoordinator

AgentCoordinator remains the public analysis boundary. Existing callers still import `AgentCoordinator`, call `run()` or `run_sync()`, and read `AgentCoordinator/cache/coordinator_output_latest.json`. Internally, the current endpoint implementation uses `AgentCoordinator/intelligence/` as a shared EvidenceGraph substrate, then exports the long-standing Coordinator artifact fields as views over that substrate.

## Active Runtime

| Phase | Runtime Node | Output |
| --- | --- | --- |
| Understanding | `query_understanding` | Target entity, analysis type, key terms, and scope. |
| Planning | `retrieval_planner` | Budgeted `RetrievalTask` objects. |
| Acquisition | `source_acquisition` | Provider-normalized `NormalizedItem` records and diagnostics. |
| Quality | `quality_pipeline` | `QualityFeatures`, distinct evidence groups, evidence items, cited source excerpts, quality summary, freshness summary, and source coverage. |
| Claims | `claim_miner` | Claims backed by cited source excerpts. |
| Research loop | `adaptive_research_loop` | Claim-driven support/refute/primary-source retrieval tasks. |
| Audit | `evidence_audit` | Claim-level `AuditDecision` records. |
| Synthesis | `citation_grounded_synthesis` | Final cited insights and downgraded wording policy. |

The compatibility property `AgentCoordinator.graph` intentionally raises `RuntimeError` in the current entry point. The QueryAgent/DataBridge/deliberation/targeted-search LangGraph implementation remains in `AgentCoordinator/graph/`; `/api/coordinator/run` currently enters the EvidenceGraph-backed Coordinator path directly.

## Implementation

| Path | Purpose |
| --- | --- |
| `AgentCoordinator/coordinator.py` | Public Coordinator class that invokes the internal intelligence layer and exports the artifact. |
| `AgentCoordinator/intelligence/` | Internal EvidenceGraph-centered acquisition, quality, reasoning, audit, and synthesis layer. |
| `AgentCoordinator/intelligence/projection/report_engine_contract.py` | Projection from `CoordinatorIntelligenceArtifact` to the stable Coordinator JSON contract. |
| `AgentCoordinator/coordinator_output_schema.py` | Compatibility schema helper kept for tests and older callers; current endpoint export schema is `2.1-coordinator-intelligence`. |
| `AgentCoordinator/cache/` | Latest and timestamped runtime artifacts. Generated timestamped cache files are ignored. |

`AgentCoordinator/graph/` remains the LangGraph implementation path, while the current `/api/coordinator/run` entry point uses the EvidenceGraph-backed Coordinator path.

## Coordinator Output

`_export_coordinator_output()` writes:

| File | Purpose |
| --- | --- |
| `AgentCoordinator/cache/coordinator_output_<YYYYMMDD_HHMMSS>.json` | Timestamped runtime archive. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Stable latest artifact consumed by Signal Studio and ReportEngine. |

The artifact uses `schema_version = "2.1-coordinator-intelligence"` and contains `coordinator_intelligence` as the internal evidence ledger. Compatibility fields such as `synthesis`, `source_data`, `fact_opinion_separation`, `divergence_matrix`, and `deliberation` are evidence-derived views. The `artifact_derivation` field records that relationship so downstream adapters do not treat them as independent conclusions.

## Progress Reporting

Flask passes a progress callback to `AgentCoordinator.run_sync()`. The UI timeline maps the current endpoint implementation nodes:

| Node | UI Meaning |
| --- | --- |
| `query_understanding` | Intent and scope. |
| `retrieval_planner` | Retrieval budget and task plan. |
| `source_acquisition` | Search/provider acquisition. |
| `quality_pipeline` | Canonical clustering and quality scoring. |
| `claim_miner` | Source-span claim mining. |
| `adaptive_research_loop` | Follow-up retrieval decisions. |
| `evidence_audit` | Claim-level audit. |
| `citation_grounded_synthesis` | Final cited insight construction. |

## Runtime Diagnostics

| Condition | Behavior |
| --- | --- |
| Missing semantic provider | Artifact records `jina:not_configured`; deterministic rules remain visible as the route. |
| Configured provider failure | Artifact records `provider:error` with the provider exception. |
| External source acquisition unavailable | Local replay is disabled by default; when explicitly enabled with `COORDINATOR_ALLOW_REPLAY_FALLBACK=true`, the artifact records `local_fixture:used`. |
| Unsupported strong claim | Claim is weakened or rejected before synthesis. |

## Related Documents

- [QueryEngine](query-engine.md)
- [Configuration](../reference/configuration.md)
- [Runtime Flow](../architecture/runtime-flow.md)
