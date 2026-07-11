# QueryEngine

QueryEngine remains the evidence and stance component in the project architecture. The current `/api/coordinator/run` endpoint does not instantiate the QueryEngine LangGraph directly; AgentCoordinator carries the active query-layer responsibilities through `AgentCoordinator/intelligence/` and keeps QueryAgent-shaped compatibility fields in the Coordinator artifact. The QueryEngine graph, tools, evaluation utilities, provider connectors, MindSpider integration ideas, stance/coverage concepts, and output contract remain part of the codebase and design reference.

The current Coordinator endpoint runs these query-layer responsibilities inside the Coordinator intelligence layer:

```text
query_understanding
-> retrieval_planner
-> source_acquisition
-> quality_pipeline
-> claim_miner
-> adaptive_research_loop
-> evidence_audit
-> citation_grounded_synthesis
```

## Runtime Mapping

| Former QueryEngine Responsibility | Active Runtime Location |
| --- | --- |
| Query planning | `AgentCoordinator/intelligence/reasoning/planner.py` |
| Search provider dispatch | `AgentCoordinator/intelligence/acquisition/source_gateway.py` |
| Deduplication | `AgentCoordinator/intelligence/quality/pipeline.py` canonical clustering |
| Trust scoring | `QualityFeatures.source_authority_score` and `persuasiveness_score` |
| Stance classification | `QualityFeatures.stance`, `sentiment`, and `aspect` |
| Gap filling | `AgentCoordinator/intelligence/reasoning/adaptive_loop.py` |
| Output assembly | `CoordinatorIntelligenceArtifact` plus ReportEngine projection |

## Existing QueryEngine Implementation

| Path | Status |
| --- | --- |
| `QueryEngine/agent.py` | QueryAgent entry points and standalone structured research path. |
| `QueryEngine/graph/builder.py` | QueryEngine LangGraph topology. |
| `QueryEngine/graph/nodes/` | Planner/search/dedup/trust/stance/enrichment nodes. |
| `QueryEngine/evaluation/` | Existing evaluation utilities. Evaluation is not part of the current refactor acceptance gate. |

## Active Query Quality Contract

The active runtime emits these fields inside `coordinator_intelligence`:

| Field | Purpose |
| --- | --- |
| `evidence_graph.normalized_items` | Provider-normalized source items. |
| `evidence_graph.quality_features` | Relevance, informativeness, authority, freshness, stance, sentiment, aspect, coordination, and persuasiveness features. |
| `evidence_graph.canonical_clusters` | Internal distinct source groups and repeated-coverage counts. |
| `quality_summary` | Raw source counts, distinct evidence counts, duplicate ratios, low-quality ratios, and quality warnings. |
| `freshness_summary` | Newest/oldest published timestamps, median age, and stale-source ratio. |
| `source_coverage` | Web domain counts, observable social-platform counts, MindSpider sample availability, replay fixture counts, and the active coverage mode. |
| `source_coverage_limitations` | Explicit limits such as query-time search, no firehose, and local replay usage. |

## Provider Dependencies

| Layer | Providers |
| --- | --- |
| Source acquisition | Tavily, Bocha, or Anspire for web sources; optional MindSpiderDB for platform samples; optional local replay fixture. |
| Semantic quality | Jina embeddings and rerank, with deterministic rules when Jina is not configured. |
| Structured reasoning | Existing QueryEngine LLM settings: `QUERY_ENGINE_API_KEY`, `QUERY_ENGINE_BASE_URL`, and `QUERY_ENGINE_MODEL_NAME`. |

MindSpiderDB is not a presentation-only feature. When `COORDINATOR_ENABLE_MINDSPIDER_DB=true`, the planner emits a `target_source=mindspider_db` retrieval task. Returned platform samples are normalized, clustered, scored, audited, and cited through the same `QualityPipeline` as web search results. External social-platform hits from web search, such as Reddit or X/Twitter, are also normalized to platform keys, but provider diagnostics still distinguish them from MindSpiderDB samples. The Coordinator artifact then exposes the platform view through `source_data.query_agent.social_sentiment`, `platform_interpretations`, `divergence_matrix`, and `bias_analysis`.

See [Configuration](../reference/configuration.md) and [AgentCoordinator](agent-coordinator.md).
