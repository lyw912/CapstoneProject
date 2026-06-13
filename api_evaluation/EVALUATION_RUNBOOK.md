# Evaluation Runbook

Use this runbook for the current selection round:

- Chinese media/web retrieval: Bocha.
- LLM candidates: DeepSeek Chat, DeepSeek Reasoner, Qwen Plus by default.
- Optional finalists: Qwen Max, Kimi, Gemini proxy.

## 1. Set Keys

PowerShell:

```powershell
cd D:\huang\Desktop\Project\api_evaluation

$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:DASHSCOPE_API_KEY="your_qwen_key"
$env:BOCHA_WEB_SEARCH_API_KEY="your_bocha_key"
```

If you enable optional providers in `providers.local.json`, also set:

```powershell
$env:MOONSHOT_API_KEY="your_kimi_key"
$env:AIHUBMIX_API_KEY="your_aihubmix_key"
```

## 2. Quick Smoke Run

Use one repetition first to verify credentials and response compatibility.

```powershell
python run_evaluation.py --providers providers.local.json --out results_smoke_real
```

Open:

```powershell
notepad results_smoke_real\summary.csv
notepad results_smoke_real\manual_review.csv
```

If a provider has `success_rate=0`, check `raw_results.jsonl` or `manual_review.csv` error column.

## 3. More Reliable Run

Run each case three times. This is slower and costs more, but exposes instability.

```powershell
python run_evaluation.py --providers providers.local.json --out results_full_r3 --repetitions 3
```

## 4. What to Review Manually

Automatic scoring is useful for filtering bad candidates, but final selection must inspect outputs.

Review `manual_review.csv` and fill:

- `manual_task_success_0_5`
- `manual_grounding_0_5`
- `manual_format_0_5`
- `manual_notes`

Prioritize these rows:

- `query_engine` cases with JSON output.
- `report_engine` long-form report and revision cases.
- `media_engine` conflict/noise cases.
- Search rows where Bocha has low relevance or poor source diversity.

## 5. Decision Rules

Recommended defaults after review:

- MediaEngine LLM: choose the model with best grounding on `media_*` cases.
- QueryEngine LLM: choose the model with best JSON validity and stance/gap accuracy.
- ReportEngine LLM: choose the model with best long-form structure and conservative wording.
- ForumEngine LLM: choose the fastest acceptable model.
- MindSpider LLM: choose the cheapest model that keeps valid JSON and good clustering.
- Search: keep Bocha if relevance/source diversity are acceptable on Chinese cases; otherwise compare with Anspire later.

## 6. Optional Second Round

After first run, enable `qwen-max-compatible` if Qwen Plus is close but not strong enough for ReportEngine:

```json
"enabled": true
```

Then rerun with a new output folder:

```powershell
python run_evaluation.py --providers providers.local.json --out results_with_qwen_max --repetitions 3
```
