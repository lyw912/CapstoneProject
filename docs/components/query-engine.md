# QueryEngine

QueryEngine is the active breadth and stance specialist. The `/api/coordinator/run` parent graph invokes `DeepSearchAgent.research_contribution()`, which executes the QueryEngine LangGraph and emits a typed `QueryContribution` containing sources, independent acquisition observations, addressable evidence excerpts, stance coverage, opinion-cluster claim proposals, gaps, trace, and errors.

The active Query subgraph remains:

```text
query_planner -> unified_search -> dedup_filter -> trust_scorer
-> stance_classify -> social_enrichment -> coverage_check
-> gap_filler (bounded loop) -> output_assemble -> QueryContribution
```

## Runtime Mapping

| Responsibility | Active Runtime Location |
| --- | --- |
| Stance-aware planning/search/coverage | `QueryEngine/graph/` |
| Query contribution contract | `QueryEngine/contribution.py` |
| Cross-agent canonicalization and quality | `AgentCoordinator/intelligence/evidence_core/` |
| Global follow-up routing | `AgentCoordinator/fusion/supervisor.py` |

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
| `source_coverage_limitations` | Explicit limits such as query-time search, no firehose, missing specialist configuration, or provider failure. |

## Provider Dependencies

| Layer | Providers |
| --- | --- |
| Source acquisition | Tavily, Bocha, or Anspire for web sources; optional read-only MindSpiderDB platform samples. |
| Semantic quality | Jina embeddings and rerank, with deterministic rules when Jina is not configured. |
| Structured reasoning | Existing QueryEngine LLM settings: `QUERY_ENGINE_API_KEY`, `QUERY_ENGINE_BASE_URL`, and `QUERY_ENGINE_MODEL_NAME`. |

MindSpiderDB is optional and read-only by default. `COORDINATOR_ENABLE_MINDSPIDER_DB=true` enables QueryEngine database routes against the existing crawl tables. `COORDINATOR_ALLOW_MINDSPIDER_CRAWL_TRIGGER=false` prevents a Coordinator request from implicitly starting BroadTopicExtraction; enabling crawling is a separate operational decision. Each database hit records its query, task, provider, source table metadata, and retrieval time as an `AcquisitionObservation` before EvidenceCore canonicalization.

See [MindSpider Data Contract](../reference/mindspider-data-contract.md), [Configuration](../reference/configuration.md), and [AgentCoordinator](agent-coordinator.md).
