# Sentiment Models

`SentimentAnalysisModel/` contains model experiments and supporting datasets for sentiment analysis and topic detection. These models are project assets rather than the primary final Signal Studio runtime path.

## Model Families

| Path | Model Family | Purpose |
| --- | --- | --- |
| `SentimentAnalysisModel/WeiboSentiment_MachineLearning/` | Classical ML and neural baselines | Bayes, SVM, XGBoost, LSTM, BERT experiments for Weibo sentiment. |
| `SentimentAnalysisModel/WeiboSentiment_SmallQwen/` | Qwen-based sentiment | LoRA/embedding/predict utilities for small Qwen sentiment workflows. |
| `SentimentAnalysisModel/WeiboSentiment_Finetuned/BertChinese-Lora/` | BERT LoRA | Chinese sentiment fine-tuning and prediction. |
| `SentimentAnalysisModel/WeiboSentiment_Finetuned/GPT2-Lora/` | GPT-2 LoRA | Sentiment fine-tuning and prediction. |
| `SentimentAnalysisModel/WeiboSentiment_Finetuned/GPT2-AdapterTuning/` | GPT-2 adapter tuning | Adapter-based fine-tuning experiments. |
| `SentimentAnalysisModel/WeiboMultilingualSentiment/` | Multilingual sentiment | Prediction utilities for multilingual sentiment. |
| `SentimentAnalysisModel/BertTopicDetection_Finetuned/` | Topic detection | Fine-tuned BERT topic classifier. |

## Data Assets

| Dataset Path | Notes |
| --- | --- |
| `WeiboSentiment_MachineLearning/data/weibo2018/` | Train/test data and topic files. |
| `WeiboSentiment_MachineLearning/data/stopwords.txt` | Stopword resource. |
| `WeiboSentiment_SmallQwen/dataset/weibo_senti_100k.csv` | Weibo sentiment dataset. |
| `WeiboSentiment_Finetuned/*/dataset/` | Fine-tuning datasets. |
| `BertTopicDetection_Finetuned/dataset/` | Topic-detection train/valid/test CSVs. |

## Runtime Relationship

| Relationship | Current State |
| --- | --- |
| Final Signal Studio path | Uses LLM/search engines and Coordinator artifacts; does not require these local model experiments for a standard run. |
| Research value | Provides alternative sentiment/topic modeling options and reproducible model work. |
| Documentation status | Operational summaries are centralized here; model-specific READMEs remain near code and datasets. |

## Operational Notes

| Concern | Recommendation |
| --- | --- |
| Large model files | Avoid moving or renaming model artifacts unless code paths are updated. |
| Dependency conflicts | Use model-family-specific requirements files when running experiments. |
| Dataset privacy | Treat raw social datasets as project data assets; do not publish sensitive data inadvertently. |
| Integration | If integrating into QueryEngine social enrichment, define a stable adapter and document it in [QueryEngine](query-engine.md). |

See [Data And Model Assets](../reference/data-and-model-assets.md) for clone impact, publication, licensing, and privacy guidance.
