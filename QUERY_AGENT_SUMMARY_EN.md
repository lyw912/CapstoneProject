# Query Agent v2.0 — Development Summary & Collaboration Guide

> **Author:** Member A | **Updated:** 2026-04-06
> **Target Audience:** Team members looking to understand QueryAgent's implementation, operation, and integration.

---

## I. Upgrades & Enhancements

### 1.1 Comparison with Original QueryEngine

The original `QueryEngine/agent.py` was essentially a fixed pipeline script. The upgraded **v2.0** is a true autonomous Agent.

| Dimension             | Original QueryEngine            | Upgraded Query Agent v2                                                   |
| :-------------------- | :------------------------------ | :------------------------------------------------------------------------ |
| **Architecture**      | Fixed linear pipeline (6 nodes) | **LangGraph Subgraph** (8 nodes with conditional loops)                   |
| **Search Strategy**   | Single source (Tavily)          | **Multi-source Parallel** (Tavily + Anspire + InsightDB)                  |
| **Stance Awareness**  | None                            | **5D Stance Matrix** (Official / Support / Oppose / Neutral / Background) |
| **Source Evaluation** | Equal weighting                 | **TrustScore 4D Scoring** (Authority, Timeliness, Quality, Rank)          |
| **Deduplication**     | None                            | **URL Exact** + **MinHash LSH** Content Deduplication                     |
| **Termination**       | Fixed rounds                    | **SCS-driven Adaptive Termination** (Stance Coverage Score)               |
| **Output Format**     | Unstructured Markdown           | **Structured `QueryAgentOutput` (JSON)**                                  |

### 1.2 New Directory Structure (Original files remain untouched)

```text
QueryEngine/
├── graph/                      ← All Newly Added
│   ├── state.py                # LangGraph State Definition (TypedDict)
│   ├── builder.py              # Graph Construction (incl. coverage_router)
│   └── nodes/                  # 8 Node Implementations
│       ├── query_planner.py
│       ├── unified_search.py
│       ├── dedup_filter.py
│       ├── trust_scorer.py
│       ├── stance_classify.py
│       ├── coverage_check.py
│       ├── gap_filler.py
│       └── output_assemble.py
├── classifiers/                ← All Newly Added
│   ├── trust_scorer.py         # 4D TrustScore Implementation
│   └── stance_classifier.py    # Hybrid Rule + Keyword Classification
├── fusion/                     ← All Newly Added
│   ├── rrf.py                  # Reciprocal Rank Fusion (SIGIR 2009)
│   └── dedup.py                # MinHash LSH Deduplication
├── tools/
│   └── search_dispatcher.py    ← New: Unified Dispatch for Tavily/Anspire/InsightDB
└── evaluation/                 ← All Newly Added
    ├── metrics.py              # SCS/SDI/SBS/TSM Calculations
    ├── test_queries.py         # 20 Standard Test Queries
    └── run_evaluation.py       # CLI Evaluation Script
```

`agent.py` now includes `research_structured()`, `research_structured_sync()`, `query_graph` properties, and `_write_forum_finding()`. **The original `research()` method is fully preserved.**

### 1.3 Core Algorithm Optimizations

#### Authoritative Source Priority Deduplication

- **Implementation File**: [dedup_filter.py](file:///e:/GitHubdesk/CapstoneProject/QueryEngine/graph/nodes/dedup_filter.py)
- **Optimization Point**: The system now recognizes official domains (e.g., `.gov.cn`, `xinhua.net`). During URL and content deduplication, if similar content exists, it **prioritizes retaining official or authoritative sources** to prevent high-authority information from being discarded due to duplicate entries from secondary sites.

---

## II. Innovation Points & Literature Support

### 2.1 Comparison with Existing Systems

| System               | Query Decomposition      | Stance Awareness   | Closed-loop Termination   | Trustworthiness Eval |
| :------------------- | :----------------------- | :----------------- | :------------------------ | :------------------- |
| GPT-Researcher       | what/why/how dimensions  | ❌                 | ❌ Fixed Rounds           | ❌                   |
| STORM (ACL 2024)     | Expert Roles (Knowledge) | ❌ Role ≠ Stance   | ❌                        | ❌                   |
| MindSearch (2024)    | DAG Logical Sub-queries  | ❌                 | ❌                        | ❌                   |
| Self-RAG (ICLR 2024) | No Decomposition         | ❌                 | Relevance Self-reflection | ❌                   |
| **Query Agent v2**   | **5D Stance Matrix**     | **✅ Core Design** | **✅ SCS Driven**         | **✅ TrustScore**    |

### 2.2 Innovation 1: Stance Matrix Sub-query Planning

Existing Deep Research Agents do not consider stance diversity during the **query generation phase**. This solution is unique in injecting stance constraints at the planning stage (solving diversity upstream rather than via downstream reranking).

```python
# QueryEngine/graph/nodes/query_planner.py
# Force coverage of 5 stance dimensions during LLM generation
# Fallback: _ensure_stance_coverage() ensures all dimensions are represented
```

**Literature Basis**: Draws et al. (SIGIR 2021) — Stance bias measurement; MMR/xQuAD (Diversity solved upstream).

### 2.3 Innovation 2: Closed-loop Stance Coverage Check

While Self-RAG / CRAG reflection loops evaluate **information relevance**, our reflection loop evaluates **stance coverage** (a brand-new dimension).

```text
SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
Thresholds = {support:2, oppose:2, official:1, neutral:1}

SCS < 1.0 AND rounds < 3  →  GapFiller generates follow-up search  →  Return to Search
SCS = 1.0 OR rounds = 3   →  Final Output
```

**Literature Basis**: Self-RAG (ICLR 2024), CRAG (arXiv 2401.15884), Adaptive-RAG (arXiv 2403.14403).

### 2.4 Innovation 3: Multi-dimensional Trustworthiness (TrustScore)

```python
# QueryEngine/classifiers/trust_scorer.py
score = 0.30 * domain_authority    # 60+ authoritative domain dictionary
      + 0.25 * timeliness          # 7-day half-life exponential decay
      + 0.25 * content_quality     # snippet length + full-text availability
      + 0.20 * rrf_score           # Search API relevance score
```

---

## III. System Architecture

### 3.1 Query Agent Internal Execution Flow

```text
User Query (str)
    │
    ▼ query_planner
    │  LLM generates 5-8 sub-queries covering 5 stance dimensions
    │  official → Inject official domain filters (.gov.cn, xinhua.net, etc.)
    │
    ▼ unified_search (asyncio.gather parallel)
    │  target_source=tavily    → TavilyNewsAgency (International News)
    │  target_source=anspire   → AnspireAISearch (Chinese Media)
    │  target_source=insight_db → MediaCrawlerDB (MindSpider Social Media Data)
    │
    ▼ dedup_filter
    │  URL Exact Dedup → MinHash LSH Content Dedup (Jaccard ≥ 0.8)
    │
    ▼ trust_scorer
    │  Calculate trust_score ∈ [0, 1] for each source
    │
    ▼ stance_classify
    │  Rules (0.90) > Keywords (0.50-0.85) > Sub-query Labels (0.50) > Default Neutral (0.40)
    │
    ▼ coverage_check
    │  Calculate SCS, identify missing_stances
    │
    ├─ Sufficient Coverage ──────────────────────────┐
    ├─ Reached 3 Rounds Limit ───────────────────────┤
    └─ Missing Stances → gap_filler (LLM generates follow-up) │
                    └→ Return to unified_search      │
                                                 ▼
                                         output_assemble
                                         Stance Dist + OpinionCluster(LLM)
                                         + KnowledgeGaps(LLM)
                                         → QueryAgentOutput (dict)
```

### 3.2 QueryAgentOutput Data Structure

```python
# QueryEngine/graph/state.py — QueryAgentOutput TypedDict
{
    "original_query":     str,
    "analysis_type":      str,              # event/brand/policy/person/general
    "search_iterations":  int,              # Actual search rounds (1-3)
    "total_sources_found": int,             # Count before deduplication
    "total_sources_kept":  int,             # Count after deduplication
    "stance_distribution": {               # Proportion per stance (0-1)
        "support": 0.30, "oppose": 0.20,
        "official": 0.15, "neutral": 0.25, "background": 0.10
    },
    "opinion_clusters": [                  # LLM clustering per stance
        {
            "stance":               "oppose",
            "core_argument":        "Core argument (1 sentence)",
            "representative_quote": "Representative original quote",
            "source_count":         5,
            "estimated_proportion": 0.20,
        }, ...
    ],
    "sources": [                           # SourceItem list (descending TrustScore)
        {
            "url": str, "title": str, "snippet": str,
            "source_api": "tavily"/"anspire"/"insight_db",
            "trust_score": 0.73,           # 0-1
            "stance_label": "oppose",      # Stance label
            "stance_confidence": 0.80,
            "platform": "reuters.com",
        }, ...
    ],
    "knowledge_gaps":    ["Unresolved questions...", ...],
    "coverage_score":    0.875,            # SCS value
    "structured_summary": "",             # Phase 3 Implementation
    "trace_log":         [...],
}
```

### 3.3 Relationship with the Entire System

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Entire System Architecture                  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Information Acquisition Layer                            │    │
│  │                                                           │    │
│  │  MindSpider ──Crawling 7 Social Platforms──→ MySQL       │    │
│  │      ↑ Independent Background Process    ↑                 │    │
│  │                          InsightEngine.MediaCrawlerDB    │    │
│  │                                      ↑ Optional 3rd Source│    │
│  │  [Query Agent v2]──────────────────────────────────┐     │    │
│  │  Tavily (Intl) + Anspire (CN) + InsightDB (Optional) │     │    │
│  │  → Stance Matrix Planning → Multi-source Parallel Search │    │
│  │  → Dedup + Scoring + Classification                      │     │
│  │  → QueryAgentOutput (Structured JSON)                    │     │
│  └──────────────────────────────────────────────────────────┘    │
│                         ↓ Text output logs                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Collaboration Layer (ForumEngine)                       │    │
│  │                                                           │    │
│  │  query.log ──┐                                           │    │
│  │  media.log ──┼→ LogMonitor → forum.log ←→ Frontend UI    │    │
│  │              └→ Every 5 messages → ForumHost LLM → Summary│    │
│  │                                 ↓                         │    │
│  │              SummaryNode reads latest HOST summary         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  MediaEngine (Bocha CN Search)    InsightEngine (MySQL Query)      │
│  ReportEngine (Markdown Report)    Flask Main App (Orchestration)   │
└─────────────────────────────────────────────────────────────────┘
```

**Information flow of the three engines (independent of each other, search results are not shared):**

| Engine                           | Data Source                  | Writing            | Reading                  |
| :------------------------------- | :--------------------------- | :----------------- | :----------------------- |
| QueryEngine (Original process)   | Tavily                       | query.log          | forum.log (HOST message) |
| **Query Agent v2 (New process)** | Tavily + Anspire + InsightDB | Not yet integrated | —                        |
| MediaEngine                      | Bocha                        | media.log          | forum.log                |
| InsightEngine                    | MySQL                        | insight.log        | forum.log                |

---

## IV. How Other Team Members Integrate

### 4.1 Invocation Method

```python
from QueryEngine.agent import DeepSearchAgent

agent = DeepSearchAgent()

# Asynchronous call (Recommended, inside an async function)
output = await agent.research_structured("DeepSeek releases new model - various opinions")

# Synchronous call (Compatible with non-async environments)
output = agent.research_structured_sync("DeepSeek releases new model - various opinions")

# output is a QueryAgentOutput dict, fields as shown in 3.2 above
```

### 4.2 Quick Evaluation

```bash
# Quick verification (Run Q01 only)
python -m QueryEngine.evaluation.run_evaluation --quick

# Specify Query IDs
python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16

# Full 20-query evaluation
python -m QueryEngine.evaluation.run_evaluation --full
```

### 4.3 Visualization Interface

```bash
streamlit run SingleEngineApp/query_agent_temp_app_2.0.py
```

Four Tabs: Stance Distribution / Source List (filterable by stance) / Opinion Clustering / Knowledge Gaps

### 4.4 ForumEngine Integration (⚠️ Currently disconnected, needs processing)

**Problem**: ForumEngine's LogMonitor monitors these patterns:

```python
# ForumEngine/monitor.py
self.target_node_patterns = [
    'FirstSummaryNode',        # ← LangGraph nodes do not output this
    '正在生成首次段落总结',     # ← LangGraph nodes do not output this
]
```

`research_structured()` uses LangGraph nodes with prefixes like `[QueryPlanner]`, `[OutputAssemble]`, etc., which are **not captured**.

**Two repair solutions** (choose one):

**Solution A (Recommended, modify agent.py)**: Call the already implemented `_write_forum_finding(output)` before `research_structured()` returns, directly writing `[FINDING]` formatted messages to `query.log`.

```python
# QueryEngine/agent.py — Add to the last few lines of research_structured():
self._write_forum_finding(output)   # This method is already implemented, just add one line call
return output
```

**Solution B (Modify ForumEngine)**: Add LangGraph node log prefixes to `target_node_patterns` in `monitor.py`:

```python
self.target_node_patterns += [
    '[QueryPlanner]', '[OutputAssemble]', '[CoverageCheck]'
]
```

### 4.5 ReportEngine Integration

ReportEngine consumes Markdown. The `_output_to_markdown()` method has not yet been implemented and needs to be supplemented. Template can be found in `QUERY_AGENT_ARCHITECTURE_v2_PART2.md` §9.1.

### 4.6 InsightEngine Using QueryAgent Results

InsightEngine can directly receive the `QueryAgentOutput` dict for secondary analysis; the interface is already stable.

QueryAgent using InsightEngine data: Just via `MediaCrawlerDB` (already implemented in the `_search_insight_db()` branch in `search_dispatcher.py`, requires MindSpider to have crawled data).

---

## V. Existing Problems and Limitations

### 5.1 Architectural Issues

| Problem                    | Severity | Description                                                                                                                   |
| :------------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **ForumEngine Disconnect** | High     | `research_structured()` logs don't match monitor patterns. QueryAgent is "silent" in collaboration. (Fix: add one-line call). |
| **Isolated Engines**       | Medium   | MediaEngine, InsightEngine, and QueryAgent do not share search results, leading to potential redundant searches.              |

### 5.2 MediaEngine Multimodal Problems

MediaEngine is named a multimodal agent but actually only handles text; images and videos are completely unused:

```python
# MediaEngine/tools/search.py
images:      List[ImageResult]      # Discarded
modal_cards: List[ModalCardResult]  # Video/Weather/Stock - Discarded
webpages:    List[WebpageResult]    # Only text is processed
```

**Improvement**: Parse Bocha's `video` type cards into text snippets (title, views, date) for social media analysis.

### 5.3 Integration of MindSpider and QueryAgent

Current integration method (Spider → MySQL → InsightEngine.MediaCrawlerDB → QueryAgent's InsightDB source). Spider is asynchronous background collection (hourly), while QueryAgent is synchronous real-time query (within minutes). **Considerable?**.

---

_Document Version: v2.0 | 2026-04-07_
