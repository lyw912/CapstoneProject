# Query Agent — Development Summary (v3.1)

> **Author:** li_yewen | **Updated:** 2026-05-02
> **Scope:** QueryEngine v2 (LangGraph rewrite) + Phase 3 MindSpider social media integration

---

## I. Project Overview

**BettaFish** is a multi-agent Chinese public opinion analysis platform forked from the open-source project `https://github.com/666ghj/BettaFish`.

### Five-Layer Architecture

```
Layer 1: UI Layer
  Flask main app (app.py, port 5000)
  + Three Streamlit sub-apps (SingleEngineApp/, ports 8501-8503)

Layer 2: Multi-Agent Analysis Layer
  QueryEngine   — Tavily web search + DeepSeek reasoning
  MediaEngine   — Bocha/Anspire Chinese multimodal search + Gemini 2.5 Pro
  InsightEngine — MySQL/PostgreSQL social media DB query + Kimi K2 (500K context)
  ForumEngine   — LLM moderator coordination (Qwen Plus), file-bus communication

Layer 3: Report Generation Layer
  ReportEngine  — Template selection → layout planning → section generation → HTML/PDF/Markdown

Layer 4: Data Collection Layer
  MindSpider    — Background crawler covering 6 social platforms
                  (XHS / Douyin / Bilibili / Weibo / Tieba / Zhihu)
                  Note: Kuaishou removed (content overlap with Douyin)

Layer 5: Storage & External Services
  MySQL (social media data, capstone database)
  Tavily / Bocha / Anspire APIs (search)
  LLM APIs (DeepSeek / Gemini / Kimi / Qwen)
```

---

## II. Query Agent v2 — LangGraph Rewrite

### 2.1 Comparison with Original QueryEngine

| Dimension | Original QueryEngine | Query Agent v2 |
|:----------|:---------------------|:---------------|
| Architecture | Fixed linear pipeline (6 nodes) | LangGraph subgraph (8 nodes + conditional loops) |
| Search Strategy | Single source (Tavily) | Multi-source parallel (Tavily + Anspire + MindSpider) |
| Stance Awareness | None | 5D Stance Matrix (Official/Support/Oppose/Neutral/Background) |
| Source Evaluation | Equal weighting | TrustScore 4D scoring (Authority, Timeliness, Quality, Rank) |
| Deduplication | None | URL exact + MinHash LSH content dedup |
| Termination | Fixed rounds | SCS-driven adaptive termination |
| Output Format | Unstructured Markdown | Structured `QueryAgentOutput` (JSON) |

### 2.2 Graph Topology (Phase 3)

```
START → query_planner → unified_search → dedup_filter → trust_scorer
  → stance_classify → social_enrichment → coverage_check → [router]
                                                             ├─ output_assemble → END
                                                             └─ gap_filler → unified_search
```

### 2.3 New Directory Structure

```
QueryEngine/
├── graph/
│   ├── state.py                # LangGraph state definition (TypedDict)
│   ├── builder.py              # Graph construction, coverage_router
│   └── nodes/
│       ├── query_planner.py    # Stance matrix sub-query planning
│       ├── unified_search.py   # Multi-source parallel search
│       ├── dedup_filter.py     # URL dedup + MinHash LSH
│       ├── trust_scorer.py     # TrustScore 4D scoring
│       ├── stance_classify.py  # Stance classification
│       ├── social_enrichment.py  # Phase 3: MindSpider integration node
│       ├── coverage_check.py   # SCS calculation + routing
│       ├── gap_filler.py       # Gap fill sub-queries
│       └── output_assemble.py  # Structured output assembly
├── classifiers/
│   ├── trust_scorer.py         # 4D TrustScore implementation
│   └── stance_classifier.py    # Hybrid rule + keyword + LLM classification
├── fusion/
│   ├── rrf.py                  # Reciprocal Rank Fusion (SIGIR 2009)
│   └── dedup.py                # MinHash LSH deduplication
├── tools/
│   ├── search_dispatcher.py    # Unified dispatch for Tavily/Anspire/MindSpider
│   └── mindspider_search.py    # Phase 3: MindSpiderDB search client
└── evaluation/
    ├── metrics.py              # SCS/SDI/SBS/TSM calculations
    ├── test_queries.py         # 20 standard test queries
    └── run_evaluation.py       # CLI evaluation script
```

### 2.4 Core Algorithms

#### Stance Matrix Sub-query Planning

LLM generates 5–8 sub-queries at planning time, each tagged with a target stance dimension. `_ensure_stance_coverage()` guarantees all 5 dimensions are represented even if the LLM misses one. This solves diversity upstream rather than via downstream reranking.

**Literature basis:** Draws et al. (SIGIR 2021) — stance bias measurement; MMR/xQuAD diversity.

#### SCS-Driven Adaptive Termination

```
SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
Thresholds = {support:2, oppose:2, official:1, neutral:1}

SCS < 1.0 AND rounds < 3  →  GapFiller generates follow-up queries  →  back to search
SCS = 1.0 OR rounds = 3   →  final output
```

**Literature basis:** Self-RAG (ICLR 2024), CRAG (arXiv 2401.15884), Adaptive-RAG (arXiv 2403.14403).

#### TrustScore

```python
score = 0.30 * domain_authority    # 60+ authoritative domain dictionary
      + 0.25 * timeliness          # 7-day half-life exponential decay
      + 0.25 * content_quality     # snippet length + full-text availability
      + 0.20 * rrf_score           # search API relevance score
```

#### Authoritative Source Priority Deduplication

During MinHash LSH deduplication, when two near-duplicate items are found, the one from an authoritative domain (`.gov.cn`, `xinhua.net`, etc.) is retained over the secondary source.

---

## III. Phase 3 — MindSpider Social Media Integration

### 3.1 Design Goals

1. **Dual-layer information acquisition**: Tavily/Anspire provides the web/media narrative layer; MindSpider provides the social media sentiment layer.
2. **Cross-Source Sentiment Difference (CSSD)**: Automatically detect divergence between web search results and social media discussions.
3. **Graceful degradation**: When MindSpider has no data, `social_sentiment` returns `null` and the pipeline runs as pure API mode — no behavior change.
4. **Source traceability**: All social media data shows platform, URL, and publish time to confirm it is not fabricated.

### 3.2 Academic References

| Technique | Source | Role in System |
|-----------|--------|----------------|
| Resource Selection | Callan et al., CORI, SIGIR 1995 | Probe-then-decide: lightweight COUNT query before committing to full search |
| Corrective RAG (CRAG) | Yan et al., 2024 | Social media data as a corrective source cross-validating web results |
| Adaptive-RAG | Jeong et al., 2024 | Dynamic strategy selection based on data availability |
| RRF | Cormack et al., SIGIR 2009 | Cross-source rank fusion (existing) |
| Stance Detection | Mohammad et al., SemEval 2016 | Stance classifier extended to cross-source comparison |

### 3.3 social_enrichment Node Logic

```
1. Extract keywords from query (split English tokens and Chinese text)
2. Probe MindSpider per keyword (COUNT query, <50ms each)
3. Determine mode:
   - total_posts < 3        → disabled (pure API mode)
   - freshness < 72h        → available (full hybrid mode)
   - freshness >= 72h       → stale (downweighted)
4. Full query: fetch social media posts
5. LLM batch stance classification (Ext 4)
6. Compute CSSD score
7. LLM generates cross-source comparison summary
8. Select top 10 representative social voices
9. Comment sentiment aggregation (Ext 1)
10. Temporal sentiment tracking (Ext 2)
11. Fire-and-forget BTE trigger if data stale/absent (Ext 3)
```

### 3.4 CSSD Formula

```
CSSD = 1 - cosine_similarity(web_stance_vector, social_stance_vector)

stance_vector = [support_ratio, oppose_ratio, neutral_ratio, official_ratio, background_ratio]

CSSD = 0: identical distributions
CSSD = 1: completely opposite
CSSD > 0.5: notable difference (system raises a warning)
```

### 3.5 Extension 1 — Comment Sentiment Aggregation

Posts are surface-level; comment sections reveal deeper sentiment. `MindSpiderDB.search_comments()` searches across all 6 comment tables, classifies comments with the LLM batch classifier, and ranks by like count.

Output field: `social_sentiment.comment_sentiment`

```json
{
  "total": 6,
  "distribution": {"support": 0.667, "neutral": 0.333},
  "top_comments": [
    {"platform": "weibo", "content": "...", "like_count": 567, "stance": "support"}
  ]
}
```

### 3.6 Extension 2 — Temporal Sentiment Tracking

`MindSpiderDB.search_with_time_buckets()` groups posts by date (7-day window). Each bucket is classified; trend direction is detected by comparing support ratio in the first half vs second half (delta > 0.1 = rising/falling). Already-classified posts are reused via a content→stance lookup map to avoid redundant LLM calls.

Output field: `social_sentiment.sentiment_trend`

```json
{
  "buckets": [{"date": "2026-05-01", "post_count": 23, "distribution": {...}}],
  "trend_direction": "stable",
  "trend_summary": "..."
}
```

### 3.7 Extension 3 — Active BroadTopicExtraction Trigger

When mode is `disabled` or `stale` and `daily_topics` has no entry for today, a detached subprocess is launched via `subprocess.Popen(start_new_session=True)` to run BroadTopicExtraction (~30s, no Playwright). The current query is not blocked.

### 3.8 Extension 4 — LLM Batch Stance Classifier

`HybridStanceClassifier.classify_batch_llm()` sends all posts in a single LLM call with numbered entries, returning a JSON array of stances. Falls back to rule-based per-post classification on failure. LLM classification detects implied stances and sarcasm that keyword matching misses.

**Effect on same 23 social posts:**

| Classifier | support | oppose | neutral | background |
|------------|---------|--------|---------|------------|
| Rule-based | 17.4% | 13.0% | 65.2% | 4.3% |
| LLM batch | 26.1% | 17.4% | 47.8% | 8.7% |

### 3.9 Quality Assurance Measures

**Bot/astroturfing detection**: Content diversity score = `unique_posts / total_posts`. If < 0.7 with ≥ 5 posts, a warning is surfaced: "Results may be influenced by coordinated posting."

**Political risk mitigation**: The comparison is framed as "web search results vs social media discussions" — never as "official vs public opinion." All LLM prompts include an explicit constraint: `Do NOT frame this as "official vs public" or imply any political narrative.`

**Per-platform breakdown**: Sentiment is computed separately per platform, exposing demographic differences (e.g., Weibo: emotional/mass-market; Zhihu: analytical/high-education; Bilibili: young/diverse).

**Performance optimization**: The `social_enrichment` node skips re-execution if `social_sentiment` is already in state (coverage loop re-runs). Temporal analysis reuses already-classified posts. Total latency: ~29.6s (down from ~73s, -59%).

### 3.10 Output Format

`QueryAgentOutput` gains a `social_sentiment` field:

```json
{
  "social_sentiment": {
    "mode": "available",
    "platforms_queried": ["weibo", "zhihu", "bilibili"],
    "total_posts": 23,
    "total_comments": 6,
    "sentiment_distribution": {"support": 0.261, "oppose": 0.174, "neutral": 0.478, "background": 0.087},
    "per_platform": {
      "weibo": {"count": 10, "distribution": {"support": 0.5, "oppose": 0.3, "neutral": 0.2}},
      "zhihu": {"count": 8, "distribution": {"neutral": 0.875, "background": 0.125}},
      "bilibili": {"count": 5, "distribution": {"support": 0.4, "oppose": 0.2, "neutral": 0.2, "background": 0.2}}
    },
    "content_diversity": 0.826,
    "low_diversity_warning": null,
    "divergence_score": 0.181,
    "divergence_summary": "Web search results lean toward supportive and background stances, while social media shows more balanced distribution with notable opposition.",
    "freshness_hours": 0.5,
    "top_social_voices": [
      {
        "platform": "weibo",
        "content": "...",
        "url": "https://m.weibo.cn/detail/5001004",
        "publish_time": "2026-05-01T20:51:56",
        "stance": "support"
      }
    ],
    "comment_sentiment": {"total": 6, "distribution": {...}, "top_comments": [...]},
    "sentiment_trend": {"buckets": [...], "trend_direction": "stable", "trend_summary": "..."},
    "crawl_triggered": false
  }
}
```

When MindSpider has no data, `social_sentiment` is `null` and system behavior is identical to pre-Phase 3.

### 3.11 Validation Results (2026-05-01)

Test query: "DeepSeek releases new model — various opinions"

| Metric | Value |
|--------|-------|
| Social media mode | available |
| Platforms | weibo, zhihu, bilibili |
| Total posts | 23 |
| Total comments | 6 |
| CSSD score | 0.181 |
| Content diversity | 82.6% (healthy) |
| Trend direction | stable |
| Total latency | 29.6s |
| Sources kept | 53 |

---

## IV. MindSpider Setup & Operation

### 4.1 Architecture

```
MindSpider/
├── BroadTopicExtraction/    # Stage 1: Topic discovery (lightweight, no login required)
└── DeepSentimentCrawling/   # Stage 2: Deep crawl (Playwright, requires cookies)
    ├── MediaCrawler/        # Actual crawler core (browser automation)
    ├── bilibili_api_crawler.py   # Bilibili: uses bilibili-api-python (bypasses 412 bot detection)
    └── zhihu_cookie_refresher.py # Zhihu: auto-refreshes z_c0 token via Playwright + SESSIONID
```

**BroadTopicExtraction**: Calls public news aggregation APIs (12 platforms), extracts keywords with DeepSeek, writes to `daily_news` and `daily_topics` tables. No login, no Playwright, ~30s, minimal resources.

**DeepSentimentCrawling**: Uses Playwright to simulate browser sessions, crawls post content and comments. Requires per-platform login cookies. Writes to `weibo_note`, `zhihu_content`, etc.

### 4.2 Database

All MindSpider data is stored in the **`capstone`** MySQL database (shared with the main project). Tables include:

- **Core tables**: `daily_news`, `daily_topics`, `topic_news_relation`, `crawling_tasks`
- **Platform content tables**: `xhs_note`, `douyin_aweme`, `bilibili_video`, `weibo_note`, `tieba_note`, `zhihu_content`
- **Comment tables**: `weibo_note_comment`, `zhihu_comment`, `bilibili_video_comment`, `douyin_aweme_comment`, `xhs_note_comment`, `tieba_comment`

Both `MindSpider/.env` and the project root `.env` are configured with `DB_NAME=capstone`.

**Deduplication**: All platform store layers use a select-then-update pattern (`note_id` / `comment_id` as unique keys). Re-crawling an existing post only updates `last_modify_ts` — no duplicate inserts.

### 4.3 Scheduled Crawl

File: `MindSpider/scheduled_run.sh`
Cron: `35 2 * * *` (daily at 02:35)
Persistent log: `MindSpider/logs/scheduled_run_YYYYMMDD.log`

```
Step 1: BroadTopicExtraction (~30s, no Playwright)
Step 2: Tier 1 platforms every day — weibo → zhihu → bilibili (--max-notes 20)
Step 3: Tier 2 platforms on odd days only — xhs → douyin → tieba (--max-notes 10)
```

Memory protection: checks available RAM before each platform, kills browser processes after each platform, uses a lock file to prevent cron overlap.

### 4.4 Cookie Management

File: `cookie.txt` (project root, not committed — contains credentials)
Format: one platform per line, e.g. `weibo=<cookie string>`

`MindSpider/DeepSentimentCrawling/platform_crawler.py` auto-injects the correct cookie into `MediaCrawler/config/base_config.py` before each crawl run. No manual config file editing needed.

Cookie validity as of 2026-05-02:

| Platform | Status | Expiry | Notes |
|----------|--------|--------|-------|
| Weibo | Valid | 2026-07-27 (~85 days) | Must use `m.weibo.cn` mobile cookies |
| Zhihu | Valid | SESSIONID: weeks; z_c0: auto-refreshed | Auto-refresh via `zhihu_cookie_refresher.py` |
| Bilibili | Valid | 2026-10-28 (~179 days) | Uses bilibili-api-python, no Playwright |
| XHS | Valid | ~30-day rolling | High bot-detection risk; crawled on odd days only |
| Douyin | Valid | 2026-06-30 (~60 days) | sid_guard field |
| Tieba | Valid | 6+ months | BDUSS long-lived |

**Weibo note**: MediaCrawler uses `m.weibo.cn` (mobile). Cookies must be obtained from `https://m.weibo.cn`, not `weibo.com` (desktop).

**Bilibili note**: Playwright headless mode triggers 412 bot detection. Uses `bilibili-api-python` direct API calls instead — fully bypasses browser detection.

**Zhihu note**: `zhihu_cookie_refresher.py` loads the Zhihu homepage via Playwright before each crawl using the long-lived `SESSIONID` to auto-refresh `z_c0`, then writes back to `cookie.txt`. No manual intervention needed.

### 4.5 Monitor Script

File: `MindSpider/monitor.py`
Cron: `7 8 * * *` (daily at 08:07)
Output: `/tmp/mindspider_alerts.log`

| Check | Threshold | Alert Tag |
|-------|-----------|-----------|
| Cookie validity (live HTTP) | Failure | `[COOKIE]` |
| Cookie expiry | ≤ 3 days | `[COOKIE]` |
| Crawl log cookie errors | 401/403/login expired | `[COOKIE_LOG]` |
| Crawl started but not finished | started without finished | `[CRAWL]` |
| Memory usage | > 85% | `[RESOURCE]` |
| CPU usage | > 90% (2s sample) | `[RESOURCE]` |
| Disk usage | > 80% | `[DISK]` |

Exit code: 0 = all clear, 1 = alerts triggered.

### 4.6 Resource Constraints

Server: 2 CPU cores, 2 GB RAM, 40 GB disk.

Playwright uses 200–400 MB per browser instance. Platforms must be crawled serially, not concurrently. Recommended: 1 platform at a time, daily schedule, `--max-notes 20` to keep each run under 10 minutes.

---

## V. Files Changed in Phase 3

| File | Change Type | Description |
|------|-------------|-------------|
| `QueryEngine/graph/state.py` | Modified | Added `"mindspider_db"` to `SubQueryItem.target_source`; added `social_sentiment`, `mindspider_mode` fields |
| `QueryEngine/graph/nodes/gap_filler.py` | Modified | `"insight_db"` → `"mindspider_db"` for support/oppose stances |
| `QueryEngine/graph/nodes/query_planner.py` | Modified | Prompt updated for mindspider_db; fallback routing uses mindspider_db |
| `QueryEngine/tools/mindspider_search.py` | New | `MindSpiderDB` with `probe()`, `search_comments()`, `search_with_time_buckets()`, `has_extraction_today()` |
| `QueryEngine/graph/nodes/social_enrichment.py` | New | Social enrichment node (all 4 extensions + quality measures) |
| `QueryEngine/classifiers/stance_classifier.py` | Modified | Added `classify_batch_llm()` and `_parse_batch_response()` |
| `QueryEngine/classifiers/trust_scorer.py` | Modified | +0.05 trust bonus for mindspider_db sources |
| `QueryEngine/graph/builder.py` | Modified | Registered social_enrichment node; rewired stance_classify → social_enrichment → coverage_check |
| `QueryEngine/graph/nodes/__init__.py` | Modified | Exported social_enrichment_node |
| `QueryEngine/tools/__init__.py` | Modified | Exported MindSpiderDB, MindSpiderResponse, MindSpiderResult, MindSpiderComment |
| `QueryEngine/agent.py` | Modified | Initial state includes mindspider_mode and social_sentiment fields |
| `QueryEngine/graph/nodes/output_assemble.py` | Modified | Output includes social_sentiment |
| `SingleEngineApp/query_agent_temp_app.py` | Modified | Added 5th tab "Social Sentiment" with all Phase 3 visualizations |
| `tests/sample_data.sql` | New | 23 sample social media posts (weibo/zhihu/bilibili) about DeepSeek for testing |
| `tests/sample_comments.sql` | New | 20 sample comments (weibo/zhihu) for comment sentiment testing |

---

## VI. Invocation & Visualization

### 6.1 Running the Agent

```python
from QueryEngine.agent import DeepSearchAgent

agent = DeepSearchAgent()

# Async (recommended)
output = await agent.research_structured("DeepSeek releases new model")

# Sync (non-async environments)
output = agent.research_structured_sync("DeepSeek releases new model")
```

### 6.2 Visualization Interface

```bash
streamlit run SingleEngineApp/query_agent_temp_app.py
```

Five tabs:
1. **Stance Distribution** — per-stance ratio bars
2. **Source List** — filterable by stance, shows trust score
3. **Opinion Clusters** — LLM-generated cluster summaries per stance
4. **Knowledge Gaps** — unresolved dimensions
5. **Social Sentiment** — MindSpider data: CSSD score, per-platform breakdown, comment sentiment, temporal trend chart, diversity warning, crawl trigger notice

### 6.3 Loading Test Data

```bash
mysql -u root -p capstone < tests/sample_data.sql
mysql -u root -p capstone < tests/sample_comments.sql
```

### 6.4 Quick Evaluation

```bash
python -m QueryEngine.evaluation.run_evaluation --quick
python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16
python -m QueryEngine.evaluation.run_evaluation --full
```

---

## VII. Known Issues & Limitations

| Issue | Severity | Description |
|-------|----------|-------------|
| ForumEngine disconnect | High | `research_structured()` logs use `[QueryPlanner]` etc., not matched by ForumEngine's monitor patterns. Fix: call `self._write_forum_finding(output)` before returning. |
| Isolated engines | Medium | QueryEngine, MediaEngine, InsightEngine do not share search results. |
| `structured_summary` empty | Low | `QueryAgentOutput.structured_summary` field not yet implemented. |
| MediaEngine multimodal unused | Low | Images and video modal cards are fetched but discarded; only text is processed. |
| MindSpider not required | — | The system runs fully without MindSpider data. Social sentiment degrades gracefully to null. |

---

## VIII. Team Contributions

| Member | GitHub | Contributions |
|--------|--------|---------------|
| li_yewen | li_yewen | Project architecture, Query Agent v2 design & implementation, Phase 3 MindSpider integration, MindSpider full-stack ops |
| MIAO Mengyu | mmy0302 | Query Agent post-optimization, English UI, English documentation |
| — | Crazyheartedddd | MediaEngine LangGraph rewrite, UI English translation |
| — | kzy1234 | app.py integration, MediaEngine optimization, bug fixes |
| — | Roselia-penguin | README maintenance |

---

*Document Version: v3.1 | 2026-05-02 | li_yewen*
