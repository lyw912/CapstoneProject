# Query, MindSpider, Media, and Evidence Fusion Evaluation Plan

## Status and Claim Boundary

This document defines deferred experiments. It does not claim that fusion improves quality yet. The local `DeepSeek API pricing` run is an engineering acceptance run used to verify control flow, provider participation, artifact generation, and ReportEngine compatibility. It is not an experiment because it has one topic, one run, and no blinded labels.

## Research Questions

| ID | Question | Primary measures |
| --- | --- | --- |
| RQ1 | Does MindSpider add non-duplicate, source-bound social evidence beyond web QueryEngine results? | social evidence recall, unique canonical items, novelty, citation precision |
| RQ2 | Does MediaEngine add useful narrative or multimodal evidence beyond QueryEngine? | dossier coverage, unique supported claims, asset usefulness, human relevance |
| RQ3 | Does EvidenceCore improve groundedness over direct specialist synthesis? | claim support precision, unsupported-claim rate, contradiction handling |
| RQ4 | Does adaptive follow-up improve coverage enough to justify latency and cost? | coverage gain, supported-claim gain, latency, API calls, estimated cost |
| RQ5 | Does the complete path improve report quality? | citation correctness, completeness, factuality, structure, analyst preference |

## System Variants

Run every topic with the same provider/model snapshot, retrieval date window, source budget, and report template.

| Variant | QueryAgent | MindSpider | MediaAgent | EvidenceCore | Purpose |
| --- | --- | --- | --- | --- | --- |
| Keyword/BM25 baseline | No; conventional retrieval | Optional corpus index | No | No | Non-agent retrieval floor |
| Single-agent RAG | Replaced by one generic RAG agent | No | No | No | Conventional LLM plus retrieval baseline |
| QueryAgent web-only | Yes | No | No | Yes | Isolate QueryAgent web research |
| QueryAgent + MindSpider | Yes | Yes | No | Yes | Measure social-source value; this is the repaired local acceptance path |
| QueryAgent + MediaAgent | Yes | No | Yes | Yes | Measure narrative/multimodal value |
| Full: QueryAgent + MindSpider + MediaAgent | Yes | Yes | Yes | Yes | Target fusion system |
| Previous intelligence path | Existing prior implementation | Existing prior implementation | No | Previous layer | Historical project comparator |

The repository evaluation record supports `fused`, `query_only`, `media_only`, and `previous_intelligence`. Add `query_mindspider` as an explicit run label when executing the study.

## Topic Set and Labels

Use 24-40 topics stratified across technology pricing, brand incidents, policy changes, disasters, entertainment, and international events. Include Chinese-only, English-only, and cross-lingual topics. Freeze topic wording and collection timestamp before running variants.

For each topic, two annotators independently label relevant sources, duplicate clusters, source types, target-conditioned stance, claim-to-span entailment, material coverage gaps, and report usefulness on a 1-5 rubric. Resolve disagreements with a third adjudicator and report Cohen's kappa or Krippendorff's alpha. Do not present LLM-as-judge scores without a human calibration subset.

## Metrics

| Layer | Metrics |
| --- | --- |
| Retrieval | Recall@10/20, nDCG@10, source-type coverage, social-source novelty, duplicate rate |
| Evidence | citation precision, citation recall, span entailment, unsupported-claim rate, contradiction recall |
| Orchestration | task success, provider failure rate, follow-up yield, budget violations, wall time, API calls |
| Report | factuality, completeness, stance balance, citation usability, structure, analyst preference |
| Efficiency | median/P95 latency, tokens, provider calls, estimated cost per completed report |

Use paired topic-level bootstrap confidence intervals and a Wilcoxon signed-rank test for ordinal human scores. Report effect sizes and per-topic failures, not only aggregate means. Run each stochastic variant at least three times if budget permits.

## Public Baselines and Dataset Use

| Resource | Defensible use | Limitation for this project |
| --- | --- | --- |
| [BEIR](https://github.com/beir-cellar/beir) | Sparse/dense/hybrid retrieval mechanics | Mostly not current Chinese public-opinion events |
| [MIRACL](https://project-miracl.github.io/) | Multilingual retrieval, including Chinese | Retrieval benchmark, not stance/report evaluation |
| [FEVER](https://fever.ai/) | Claim-evidence entailment baseline | Wikipedia-centric and English-centric |
| [AVeriTeC](https://fever.ai/dataset/averitec.html) | Web evidence and real-world claim verification | Fact-checking differs from population opinion analysis |
| [FreshQA](https://github.com/freshllms/freshqa) | Freshness-sensitive QA checks | Does not measure multi-platform opinion coverage |
| [RAGAS](https://arxiv.org/abs/2309.15217) | Automated faithfulness/relevance diagnostics | Must be calibrated against human labels |
| [STORM](https://github.com/stanford-oval/storm) | Public citation-backed research/report comparator | Knowledge-report objective differs from public-opinion monitoring |

Self-RAG, CRAG, and IRCoT are orchestration references rather than drop-in public-opinion datasets. Commercial deep-research tools are qualitative comparators, not reproducible academic baselines unless prompts, model versions, retrieval dates, and costs are controlled.

## Minimum Defense Package

Before claiming an improvement, complete at least 12 topics across four categories with `single-agent RAG`, `Query only`, `Query + MindSpider`, and `full fusion`; annotate at least 200 claims/citations; report provider failures and cost; and publish the frozen rubric, topic list, run configuration, and raw per-topic scores. Until then, describe the contribution as an implemented evidence-fusion architecture with an acceptance-tested execution path, not a proven quality improvement.
