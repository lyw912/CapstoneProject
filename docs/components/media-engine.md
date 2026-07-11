# MediaEngine

MediaEngine produces media-oriented deep research when invoked directly or through legacy compatibility utilities. The final `/api/coordinator/run` path does not require MediaEngine; multimodal research is marked `not_configured` in the Coordinator artifact unless a future adapter explicitly connects it as an evidence source.

## Implementation

| Path | Purpose |
| --- | --- |
| `MediaEngine/agent.py` | Agent class and sync/async research entry points. |
| `MediaEngine/graph/builder.py` | LangGraph topology. |
| `MediaEngine/graph/state.py` | Graph state wrapper. |
| `MediaEngine/state/state.py` | Research, paragraph, and search dataclasses. |
| `MediaEngine/graph/nodes/` | Report structure, paragraph processing, finalize report. |
| `MediaEngine/tools/search.py` | Bocha and Anspire search integrations. |
| `MediaEngine/prompts/prompts.py` | Prompt templates. |
| `SingleEngineApp/media_engine_streamlit_app.py` | Legacy Streamlit surface. |

## Graph Topology

| Node | Responsibility |
| --- | --- |
| `report_structure` | Build paragraph/report structure for the query. |
| `process_paragraph` | Run search and summarize one paragraph at a time. |
| `finalize_report` | Assemble final media report text. |

The paragraph router processes planned paragraphs sequentially or through `process_all_paragraphs` when `MEDIA_PARAGRAPH_WORKERS` is greater than one.

## Public Entry Points

| Method | Usage |
| --- | --- |
| `research_async(query, save_report=True)` | Async entry point for direct or legacy use; internally runs graph work in a thread. |
| `research(query, save_report=True)` | Synchronous research flow. |
| `create_agent(config_file=None)` | Factory function. |

## Search Providers

| Provider | Configuration |
| --- | --- |
| Bocha | `BOCHA_WEB_SEARCH_API_KEY`, `BOCHA_BASE_URL` |
| Anspire | `ANSPIRE_API_KEY`, `ANSPIRE_BASE_URL` |

MediaEngine also uses `MEDIA_ENGINE_API_KEY`, `MEDIA_ENGINE_BASE_URL`, and `MEDIA_ENGINE_MODEL_NAME` for LLM calls.

## Performance And Cache Controls

| Mechanism | Configuration | Purpose |
| --- | --- | --- |
| Parallel paragraphs | `MEDIA_PARAGRAPH_WORKERS` | Processes paragraph research concurrently. |
| Sequential recovery pass | `MEDIA_PARAGRAPH_RETRY_PASSES` | Reprocesses paragraphs that need another pass after parallel execution. |
| Reflection context cap | `MEDIA_REFLECTION_STATE_MAX_CHARS` | Keeps reflection prompts bounded. |
| Search timeout | `MEDIA_SEARCH_HTTP_TIMEOUT` | Controls Bocha/Anspire search request duration. |
| Legacy Coordinator budget | `COORDINATOR_MEDIA_AGENT_TIMEOUT` | Historical timeout used by the legacy Coordinator path. |
| Topic cache | `AgentCoordinator/cache/media_agent_<hash>.md` | Reuses MediaEngine Markdown output for matching topics. |

## Runtime Boundary

| Situation | Expected Behavior |
| --- | --- |
| Final Coordinator path | Multimodal evidence is not required and is projected as `media_agent.available=false` unless configured later. |
| Direct MediaEngine invocation | Runs the MediaEngine graph and writes its own output. |
| Legacy cache files | `AgentCoordinator/cache/media_agent_<hash>.md` files are runtime cache artifacts and are ignored. |

## Related Documents

- [AgentCoordinator](agent-coordinator.md)
- [Configuration](../reference/configuration.md)
- [Testing](../quality/testing.md)
