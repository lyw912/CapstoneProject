# Data And Model Assets

The repository contains sentiment, topic-detection, cached-analysis, and generated-report assets in addition to the final Signal Studio runtime. This page explains their roles and publication handling.

## Asset Inventory

| Area | Approximate Role | Runtime Role |
| --- | --- | --- |
| `SentimentAnalysisModel/WeiboSentiment_MachineLearning/` | Classical ML and neural sentiment experiments, datasets, and saved models. | Research asset. |
| `SentimentAnalysisModel/WeiboSentiment_SmallQwen/` | Qwen sentiment utilities and dataset. | Research asset. |
| `SentimentAnalysisModel/WeiboSentiment_Finetuned/` | Fine-tuning experiments for BERT/GPT-style sentiment models. | Research asset. |
| `SentimentAnalysisModel/WeiboMultilingualSentiment/` | Multilingual sentiment prediction utility. | Research asset. |
| `SentimentAnalysisModel/BertTopicDetection_Finetuned/` | Topic detection experiment data and scripts. | Research asset. |
| `static/v2_report_example/` | Curated generated report examples. | Review artifact. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Cached analysis artifact for UI/report handoff. | Review and runtime artifact. |

The final live path uses external search, semantic, and LLM providers through the Coordinator runtime, then exports a Coordinator-compatible artifact for Signal Studio and ReportEngine. QueryEngine and MediaEngine assets remain part of the project architecture and can be used directly or through Coordinator adapters.

## Clone And Storage Impact

Some dataset and model files are large. Before publishing a public repository or release artifact, decide whether these assets should remain in Git, move to Git LFS, or be replaced by download instructions.

| Asset Type | Recommended Handling |
| --- | --- |
| Small source code and configs | Keep in Git. |
| Generated reports used as examples | Keep only curated examples. |
| Large model weights | Prefer Git LFS, release assets, or external storage with checksums. |
| Large raw datasets | Prefer external storage plus documented source/license. |
| Runtime caches | Do not commit new run outputs except intentional sample fixtures. |

## License And Privacy Checklist

Before publishing data/model assets, verify:

| Check | Required Action |
| --- | --- |
| Dataset source | Document origin, license, and redistribution terms. |
| Personal data | Remove or anonymize sensitive records before public release. |
| Model weights | Confirm redistribution rights for base and fine-tuned weights. |
| Generated artifacts | Ensure they do not contain private keys, private prompts, or private user feedback. |
| Reproducibility | Provide download URL, checksum, and expected path when assets are externalized. |

## Runtime Boundary

Do not move these runtime paths without updating code and tests:

| Path | Runtime Use |
| --- | --- |
| `ReportEngine/report_template/` | Loaded by `TEMPLATE_DIR`. |
| `ReportEngine/renderers/libs/` | Used by report rendering. |
| `static/signal-studio/` | Built frontend assets served by Flask. |
| `templates/index.html` | Flask shell for Signal Studio. |
| `AgentCoordinator/cache/coordinator_output_latest.json` | Latest analysis artifact used by Signal Studio and ReportEngine. |

See [Runtime Assets](runtime-assets.md) for the broader asset boundary.
