# Coordinator Output Schema

The Coordinator artifact is written by `AgentCoordinator/coordinator.py` through `AgentCoordinator/intelligence/projection/report_engine_contract.py::build_coordinator_output_from_artifact()`. The legacy helper in `AgentCoordinator/coordinator_output_schema.py` remains for older tests and compatibility notes, but it is not the active `/api/coordinator/run` export path.

Current schema version:

```text
2.1-coordinator-intelligence
```

## Files

| File | Purpose |
| --- | --- |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Stable latest artifact consumed by Signal Studio and ReportEngine. |
| `AgentCoordinator/cache/coordinator_output_<timestamp>.json` | Timestamped runtime archive from each Coordinator run. |

Timestamped artifacts are runtime outputs. Keep only intentional sample fixtures in Git.

## Top-Level Fields

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | string | Active contract version, currently `2.1-coordinator-intelligence`. |
| `query` | string | Analysis query after Coordinator request normalization. |
| `analysis_type` | string | Planner-derived analysis type such as `brand`, `policy`, `technology`, `event`, or `general`. |
| `generated_at` | string | UTC timestamp. |
| `pipeline_duration_seconds` | number | Coordinator runtime duration. |
| `artifact_derivation` | object | Declares `coordinator_intelligence` as the primary record and compatibility fields as derived views. |
| `coordinator_intelligence` | object | Internal EvidenceGraph ledger, provider diagnostics, trace, quality summaries, audit decisions, insights, and source coverage. |
| `divergence_matrix` | object | Pairwise TVD over eligible channel-level content-stance distributions. |
| `deliberation` | object | Compatibility view over claim-level audit, contradiction edges, consensus, and dissents. |
| `gap_filling` | object | Adaptive follow-up retrieval tasks and results. |
| `platform_interpretations` | object | Platform-aware interpretation only when observable platform samples exist. |
| `bias_analysis` | object | Echo warnings and silent-majority boundary statement. |
| `fact_opinion_separation` | object | Verified facts, opinion observations, and analytical framework notes. |
| `synthesis` | object | Final cited insights, tensions, confidence, and recommended follow-up. |
| `source_data` | object | QueryAgent/MediaAgent compatibility summary derived from the EvidenceGraph. |
| `coordinator_trace` | array | Local trace messages used by Signal Studio Monitor. |
| `agent_errors` | array | Provider or runtime diagnostics that reached error state. |

## `coordinator_intelligence`

| Field | Purpose |
| --- | --- |
| `run_id`, `query`, `mode`, `created_at` | Run identity and request context. |
| `target_entity`, `analysis_type` | Query-understanding output. |
| `evidence_graph.normalized_items` | Provider-normalized source items from Tavily, Bocha, Anspire, MindSpiderDB, or explicit replay fixtures. |
| `evidence_graph.quality_features` | Relevance, informativeness, authority, freshness, stance, sentiment, aspect, coordination, and persuasiveness features. |
| `evidence_graph.canonical_clusters` | Distinct evidence groups and repeated-coverage counts. |
| `evidence_graph.evidence_items` | Representative evidence records with cited source spans. |
| `evidence_graph.claims` | Claims mined from representative source spans. |
| `evidence_graph.audit_decisions` | Claim-level accept, weaken, reject, or needs-search decisions. |
| `evidence_graph.insights` | Final insights linked to claim ids and citation span ids. |
| `quality_summary` | Duplicate, amplification, low-quality, low-relevance, and coordination summary. |
| `freshness_summary` | Newest/oldest published timestamps, median age, and stale-source ratio. |
| `source_coverage` | Web domain counts, platform sample counts, MindSpider usage flag, replay fixture count, and coverage mode. |
| `source_coverage_limitations` | Explicit scope limits such as query-time evidence, no firehose, missing providers, or local replay use. |
| `provider_diagnostics` | Search, semantic, structured-LLM, MindSpider, and fixture diagnostics. |
| `research_trace` | Structured internal node trace. |
| `budget_summary` | Research-round and provider-call budget summary. |

## Compatibility Views

The following top-level fields exist so older UI/report consumers do not need to understand the full EvidenceGraph. They must be treated as projections from `coordinator_intelligence`, not as second-pass independent conclusions.

| Field | Derived From | Notes |
| --- | --- | --- |
| `source_data.query_agent` | EvidenceGraph, quality features, source coverage | Includes totals, stance distribution, coverage score, top sources, knowledge gaps, and social sentiment view. |
| `source_data.media_agent` | Additive Media view | Reports actual specialist run status, dossier/source/asset counts, section summaries, unresolved questions, and errors. |
| `synthesis` | Audited insights | Includes citation span ids and wording policy. |
| `deliberation` | Audit decisions and contradiction edges | Replaces the historical multi-perspective graph output with claim-level audit evidence. |
| `divergence_matrix` | Canonical clusters and content-stance labels by channel | Groups web evidence into official/institutional or web/media channels, keeps native social platforms, excludes groups below three canonical clusters, and computes TVD over Laplace-smoothed support/neutral/oppose distributions. |
| `gap_filling` | Retrieval tasks/results | Shows adaptive follow-up work triggered by weak or one-sided claims. |
| `fact_opinion_separation` | Supported claims and evidence items | Preserves source span ids for traceability. |
| `platform_interpretations` | Social-platform samples | Empty for web-only runs. |
| `bias_analysis` | Quality summary and stance entropy | States that observable samples are not population-level public opinion. |

## `source_data.query_agent`

| Field | Description |
| --- | --- |
| `derived_from` | Usually `coordinator_intelligence.evidence_graph`. |
| `total_sources`, `total_sources_found` | Raw normalized item count. |
| `total_sources_kept`, `canonical_sources` | Distinct canonical evidence count. |
| `stance_distribution` | Weighted stance distribution over canonical clusters. |
| `coverage_score` | Derived coverage score with provider-diagnostic penalties. |
| `top_sources` | Ranked sources with URL, trust score, stance, sentiment, platform, source type, citation span id, and quality warnings. |
| `opinion_clusters` | Claim groups by stance. |
| `knowledge_gaps` | Missing provider, missing platform sample, or exhausted follow-up notes. |
| `quality_summary`, `freshness_summary`, `source_coverage` | Pass-through summaries from `coordinator_intelligence`. |
| `social_sentiment` | Platform view when social samples exist; otherwise mode is disabled. |

## Consumers

| Consumer | Fields Used |
| --- | --- |
| Signal Studio Home | `synthesis.overall_confidence`, `source_data.query_agent.total_sources`, `pipeline_duration_seconds`, `synthesis.key_tensions` |
| Signal Studio Readout | `synthesis.summary`, `top_insights`, `key_tensions`, `recommended_investigation` |
| Signal Studio Proof | `coordinator_intelligence`, `source_data.query_agent`, `divergence_matrix`, `platform_interpretations`, provider diagnostics |
| Signal Studio Monitor | `coordinator_trace`, `agent_errors`, metadata, feedback, local trace summary |
| ReportEngine bridge | Full artifact projected into report input and Document IR generation |

## Compatibility Guidance

When changing this schema:

1. Update `schema_version` and `artifact_derivation`.
2. Update [API Reference](api.md), [OpenAPI YAML](openapi.yaml), and Signal Studio consumers if API payloads change.
3. Update `AgentCoordinator/intelligence/projection/report_engine_contract.py` and add tests in `tests/test_coordinator_intelligence_layer.py`.
4. Keep `tests/test_coordinator_report_bridge.py` passing for historical bridge fixtures unless the compatibility contract is intentionally retired.
5. Refresh architecture diagrams when the primary runtime path or artifact shape changes.
