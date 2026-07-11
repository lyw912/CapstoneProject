# Evidence Dashboard

This page collects the strongest verification evidence stored in the repository. It is intended for reviewers who need proof beyond architecture claims.

## Provider Benchmark Summary

The current provider benchmark is stored under `api_evaluation/results_full_r3/summary.csv`. It contains three repetitions per test case for most engine/provider combinations.

| Engine | Provider Type | Best Current Provider | Score | Success Rate | Avg Latency |
| --- | --- | --- | ---: | ---: | ---: |
| QueryEngine | LLM | `deepseek-chat` | 4.505 | 1.00 | 2.423s |
| QueryEngine | Search | `bocha` | 4.668 | 1.00 | 13.006s |
| MediaEngine | LLM | `qwen-plus-compatible` | 4.418 | 1.00 | 5.673s |
| MediaEngine | Search | `bocha` | 4.646 | 1.00 | 14.399s |
| ReportEngine | LLM | `deepseek-chat` | 4.227 | 1.00 | 5.479s |
| MindSpider | LLM | `qwen-plus-compatible` | 4.378 | 1.00 | 5.217s |
| MindSpider | Search | `bocha` | 4.709 | 1.00 | 13.840s |
| ForumEngine | LLM | `qwen-plus-compatible` | 4.269 | 1.00 | 3.235s |

## Provider Tradeoffs

| Choice | Evidence | Practical Decision |
| --- | --- | --- |
| `deepseek-chat` for QueryEngine | Highest QueryEngine LLM score with lower latency than reasoning models. | Default for evidence planning and stance work. |
| `qwen-plus-compatible` for MediaEngine | Highest MediaEngine LLM score in the stored full run. | Default when media synthesis quality matters. |
| `deepseek-chat` for ReportEngine | Highest ReportEngine score and lower latency than the alternatives in the stored run. | Default for report generation. |
| `TavilyAPI` for the active Coordinator path | Current `.env.example` and Coordinator source gateway default to Tavily; Bocha remains benchmarked and selectable. | Default search provider in `.env.example`. |

## Regression Evidence

| Area | Evidence | Extension Target |
| --- | --- | --- |
| Sensitive input safety | `tests/test_sensitive_input_filter.py` | Route-level checks around Coordinator and ReportEngine APIs. |
| Coordinator intelligence layer | `tests/test_coordinator_intelligence_layer.py` | Provider routing, local replay, MindSpider samples, Jina semantic scoring, and EvidenceGraph projection. |
| Coordinator-to-report bridge | `tests/test_coordinator_report_bridge.py` | Historical schema fixture compatibility checks. |
| Report IR sanitization | `tests/test_report_engine_sanitization.py` | Full Document IR renderer fixtures. |
| MediaAgent cache/config behavior | `tests/test_media_agent_node_optional.py` | Additional timeout and cache-path scenarios. |
| Forum log parser | `tests/test_monitor.py` and fixtures | Compatibility coverage for ForumEngine log formats. |

## Runtime Evidence

| Evidence | Location | What It Proves |
| --- | --- | --- |
| Latest Coordinator artifact | `AgentCoordinator/cache/coordinator_output_latest.json` | Analysis output can be persisted in the stable schema consumed by UI and ReportEngine. |
| Static report examples | Curated generated report examples | HTML/PDF renderer output is available for artifact review. |
| Signal Studio screenshots | `docs/assets/screenshots/` | Final UI workflow is implemented and documented. |
| OpenAPI contract | `docs/reference/openapi.yaml` | Main REST surface is machine-readable. |
| Diagram sources | `docs/assets/diagrams/source/` | Architecture diagrams can be maintained and regenerated. |

## Reviewer Interpretation

The project is strongest where it combines architecture and implementation evidence: the EvidenceGraph-centered Coordinator intelligence layer, a stable Coordinator artifact, a structured ReportEngine IR, provider evaluation, and a final React operator interface. The evidence set supports a defense review of scope, implementation, API contracts, generated artifacts, and quality controls.

## Rebuild Commands

Run the benchmark when provider keys are available:

```powershell
cd api_evaluation
python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

`uv` equivalent from `api_evaluation/`:

```powershell
uv run --python 3.11 --with-requirements ../requirements.txt python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

Run the focused regression suite:

```powershell
python -m unittest tests.test_sensitive_input_filter
python -m unittest tests.test_coordinator_intelligence_layer
python -m unittest tests.test_report_engine_sanitization
python -m unittest tests.test_coordinator_report_bridge
python -m unittest tests.test_media_agent_node_optional
pytest tests/test_monitor.py -v
```

`uv` equivalent from the repository root:

```powershell
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_sensitive_input_filter
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_report_engine_sanitization
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_coordinator_report_bridge
uv run --python 3.11 --with-requirements requirements.txt python -m unittest tests.test_media_agent_node_optional
uv run --python 3.11 --with-requirements requirements.txt pytest tests/test_monitor.py -v
```

Use [Testing](testing.md) for coverage guidance and [API Evaluation](api-evaluation.md) for benchmark mechanics.
