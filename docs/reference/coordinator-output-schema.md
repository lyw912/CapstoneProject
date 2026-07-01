# Coordinator Output Schema

The Coordinator artifact is written by `AgentCoordinator/coordinator_output_schema.py::build_coordinator_output()`.

Current schema version:

```text
1.0
```

## Files

| File | Purpose |
| --- | --- |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Stable latest artifact. |
| `AgentCoordinator/cache/coordinator_output_<timestamp>.json` | Timestamped archive. |

## Top-Level Fields

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | string | Contract version. |
| `query` | string | Original analysis query. |
| `analysis_type` | string | Analysis type, usually `general`. |
| `generated_at` | string | UTC timestamp. |
| `pipeline_duration_seconds` | number | Coordinator runtime duration. |
| `divergence_matrix` | object | Pairwise divergence and hotspots. |
| `deliberation` | object | Multi-perspective reasoning output. |
| `gap_filling` | object | Search gap and supplementary retrieval summary. |
| `platform_interpretations` | object | Platform-specific interpretations. |
| `bias_analysis` | object | Echo-chamber warnings and silent-majority hypothesis. |
| `fact_opinion_separation` | object | Facts, opinions, and frameworks. |
| `synthesis` | object | Final executive synthesis. |
| `source_data` | object | QueryAgent and MediaAgent source summaries. |
| `coordinator_trace` | array | Local trace messages. |
| `agent_errors` | array | Captured agent diagnostics. |

## `divergence_matrix`

| Field | Type | Description |
| --- | --- | --- |
| `pairs` | object | String pair key to divergence value. |
| `hotspots` | array | High-divergence descriptions. |
| `max_divergence` | object | `{pair, value}` with highest divergence. |
| `min_divergence` | object | `{pair, value}` with lowest divergence. |

## `deliberation`

| Field | Type | Description |
| --- | --- | --- |
| `analysis_type` | string | Analysis type from synthesis context. |
| `perspectives_used` | array | Perspective names used in independent analysis. |
| `phases` | array | Deliberation phases with summaries, consensus, dissent. |
| `final_consensus` | array | Final consensus points. |
| `final_dissents` | array | Final dissent points. |
| `confidence` | number | Deliberation or synthesis confidence. |

## `gap_filling`

| Field | Type | Description |
| --- | --- | --- |
| `rounds_performed` | number | Search rounds triggered by gap detection. |
| `gaps_detected` | array | Objects with `description` and `source`. |
| `results_found` | number | Count of supplementary results. |

## `synthesis`

| Field | Type | Description |
| --- | --- | --- |
| `summary` | string | Executive synthesis text. |
| `top_insights` | array | Key insight objects. |
| `key_tensions` | array | Tension objects. |
| `overall_confidence` | number | Confidence score. |
| `recommended_investigation` | array | Follow-up questions/actions. |

## `source_data`

| Field | Type | Description |
| --- | --- | --- |
| `query_agent.total_sources` | number | Count of top sources considered. |
| `query_agent.stance_distribution` | object | Stance label to value. |
| `query_agent.coverage_score` | number | Evidence coverage score. |
| `query_agent.top_sources` | array | Top source objects with `title`, `url`, `trust_score`, `stance`. |
| `query_agent.social_sentiment` | object/null | Social sentiment summary when enrichment is configured. |
| `media_agent.available` | boolean | Whether MediaAgent text is available. |
| `media_agent.mode` | string | `live` or `test_data`. |
| `media_agent.summary_length` | number | Length of media text. |

## Consumers

| Consumer | Fields Used |
| --- | --- |
| Signal Studio Home | `synthesis.overall_confidence`, `source_data.query_agent.total_sources`, `pipeline_duration_seconds`, `synthesis.key_tensions` |
| Signal Studio Readout | `synthesis.summary`, `top_insights`, `key_tensions`, `recommended_investigation` |
| Signal Studio Proof | `source_data.query_agent`, `divergence_matrix`, `platform_interpretations` |
| Signal Studio Monitor | `coordinator_trace`, `agent_errors`, metadata, feedback |
| ReportEngine bridge | Full artifact converted into report inputs |

## Compatibility Guidance

When changing this schema:

1. Update `schema_version`.
2. Update [API Reference](api.md) if API payloads change.
3. Update Signal Studio consumers.
4. Update `AgentCoordinator/utils/report_bridge.py`.
5. Add or update tests under `tests/test_coordinator_report_bridge.py`.
