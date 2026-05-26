# 6. Third Update: Integration, Interim Report & Results

> Owner: Miao (mmy0302)
> Scope: End-to-end system integration, full-pipeline validation, report generation quality assessment, interim results documentation
> Status: All components integrated — end-to-end pipeline operational with validated output quality

---

## 6.1. Integration Overview

The third update marks the transition from component-level development to system-level integration. With all five major subsystems — Query Engine, Media Engine, AgentCoordinator, ForumEngine, and ReportEngine — reaching functional maturity, the focus shifted to wiring them together into a single end-to-end pipeline.

| Subsystem | Role in Pipeline | Integration Status |
|---|---|---|
| **Query Engine** | Web-sourced evidence acquisition via Tavily + DeepSeek reasoning | Integrated, producing structured evidence JSON |
| **Media Engine** | Multimodal cross-platform content analysis (TikTok, Kuaishou, Xiaohongshu) via Anspire/Bocha search | Integrated, producing media analysis reports |
| **AgentCoordinator** | Multi-perspective deliberation, CSSD divergence matrix, fact-opinion separation, CRAG gap filling | Integrated, consuming both agent outputs and producing coordinator synthesis |
| **ForumEngine** | Multi-agent discussion forum enabling structured debate between Query Agent, Media Agent, and Host | Integrated, forum log streamed to frontend in real time |
| **ReportEngine** | Template-driven report generation with interactive charts, HTML/PDF/MD export | Integrated, producing final deliverable reports |
| **Flask Orchestrator** (`app.py`) | Unified lifecycle management, subprocess orchestration, Socket.IO log streaming, REST API | Integrated, serving as the system backbone |

---

## 6.2. End-to-End Pipeline

The complete analysis workflow proceeds through the following stages:

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 1. Parallel Agent Execution                       │
│    Query Agent ──► Evidence JSON                   │
│    Media Agent ──► Media Analysis Report           │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 2. AgentCoordinator Deliberation                   │
│    · Data Bridge: normalise heterogeneous outputs  │
│    · CSSD Matrix: quantify cross-source divergence │
│    · Multi-Perspective Deliberation                │
│    · Fact-Opinion Separation                       │
│    · CRAG Gap Filling                              │
│    · Coordinator Synthesis                         │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 3. ForumEngine Multi-Agent Discussion              │
│    · Host moderates structured debate              │
│    · Agents exchange and challenge findings        │
│    · Consensus and dissent are both preserved      │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 4. ReportEngine Generation                        │
│    · Chapter-by-chapter generation via LLM         │
│    · Interactive charts (Chart.js, Sankey, etc.)   │
│    · Multi-format export: HTML / PDF / MD          │
└──────────────────────────────────────────────────┘
```

---

## 6.3. Integration Achievements

### 6.3.1. Unified Frontend Control Panel

A single-page Flask application (`templates/index.html`) serves as the unified operational dashboard:

- **Start/Stop all subsystems** with a single button
- **Real-time log streaming** from all subprocesses via Socket.IO
- **LLM configuration** with bidirectional `.env` sync (no restart required)
- **Embedded Streamlit views** for Media Agent (:8502) and Query Agent (:8503)
- **Forum chat** displaying multi-agent conversation in real time
- **Report preview** with one-click HTML/PDF/MD download

### 6.3.2. Subprocess Orchestration

`app.py` manages the full lifecycle of three child processes and one background thread:

- Streamlit subprocess for Media Agent (port 8502)
- Streamlit subprocess for Query Agent (port 8503)
- ForumEngine background thread
- ReportEngine Flask blueprint

All processes are started with health-check polling, port conflict resolution, and graceful shutdown with forced termination fallback.

### 6.3.3. Configuration Management

The LLM configuration panel achieves true bidirectional `.env` synchronisation:

- Reads current values via `GET /api/config`
- Writes updates via `POST /api/config` with immediate disk persistence
- Configuration is locked during system runtime to prevent hot-reload conflicts
- Supports all seven LLM endpoints (Insight, Media, Query, Report, MindSpider, ForumHost, KeywordOptimizer) plus search tool configuration

---

## 6.4. Interim Results

### 6.4.1. Report Generation

The integrated pipeline has successfully produced complete analysis reports. A representative example is the report generated on 2026-05-24 (`output/final_report_DeepSeek_latest_update_20260524_220313.html`, 2.3 MB), which demonstrates:

- **Multi-chapter structure** with coherent narrative flow
- **Interactive data visualisations** (charts rendered via Chart.js)
- **Source-cited evidence** with traceable provenance
- **Cross-source divergence analysis** with CSSD metrics
- **Fact-opinion separation** with clear labelling
- **Bias and echo chamber warnings** where detected

### 6.4.2. Output Formats

All three export formats have been validated:

| Format | Status | Notes |
|---|---|---|
| **HTML** | Working | Full interactive report with embedded charts and styling |
| **PDF** | Working | Server-side rendering via WeasyPrint; requires Pango for full CJK support |
| **Markdown** | Working | Raw source for archival and further processing |

### 6.4.3. Pipeline Throughput

The full pipeline — from user query to downloadable report — completes within a reasonable timeframe for a research-grade analysis system:

- Agent parallel execution: variable depending on search depth and content volume
- Coordinator deliberation: typically under 3 minutes
- Report generation: chapter-by-chapter, ~1–2 minutes per chapter
- Total end-to-end: approximately 8–15 minutes for a standard analysis

---

## 6.5. Quality Assessment

### 6.5.1. What Works Well

- **Agent complementarity**: Query Agent and Media Agent consistently produce complementary rather than redundant evidence — Query Agent excels at structured data and official sources, while Media Agent captures platform-specific sentiment and multimodal content
- **Divergence handling**: The CSSD matrix successfully identifies cases where different platforms or query strategies yield conflicting sentiment distributions, and the deliberation engine preserves these disagreements rather than collapsing them
- **Report coherence**: Despite being generated chapter-by-chapter via independent LLM calls, the final report maintains narrative coherence across chapters
- **System robustness**: The orchestrator's health-check and automatic restart logic handles transient failures gracefully

### 6.5.2. Areas for Improvement

- **Report generation time**: Chapter-by-chapter generation is inherently sequential; parallel chapter generation could reduce total wall-clock time
- **DeepSeek model constraints**: The current `deepseek-chat` model provides adequate but not exceptional reasoning quality for complex deliberation tasks; upgrading to a stronger model would improve CSSD interpretation and bias analysis
- **Report visual polish**: Chart styling and layout are functional but would benefit from design refinement

---

## 6.6. Current System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Browser (localhost:5000)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Flask-rendered SPA (index.html)                      │  │
│  │  · Search & Config · App Tabs · Console · Report      │  │
│  └──────────────────────────────────────────────────────┘  │
│  Socket.IO ── real-time log streaming                       │
└────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │       Flask app.py :5000      │
            │  · REST API (/api/*)          │
            │  · Subprocess orchestration   │
            │  · Socket.IO event bridge     │
            └───────────────┬───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Media Agent  │   │ Query Agent  │   │ ForumEngine  │
│ Streamlit    │   │ Streamlit    │   │ (background  │
│ :8502        │   │ :8503        │   │  thread)     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │  AgentCoordinator    │
                │  · Deliberation      │
                │  · CSSD Matrix       │
                │  · Fact-Opinion      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  ReportEngine        │
                │  · Template render   │
                │  · HTML/PDF/MD       │
                └──────────────────────┘
```

---

## 6.7. Conclusion

The third update confirms that all components of the multi-agent public opinion analysis system have been successfully integrated into a functioning end-to-end pipeline. The system accepts a user query, dispatches parallel analysis to the Query and Media agents, coordinates their outputs through the AgentCoordinator deliberation engine, captures agent discussion via ForumEngine, and produces a structured, evidence-traced, multi-format final report.

The temporary frontend provides full operational control and real-time observability. While visual polish and certain performance optimisations remain as future work, the core research contribution — a disagreement-aware, bias-conscious, multi-perspective public opinion analysis system — is operational and producing demonstrable results.

---

## 6.8. Next Steps

1. **Supervisor demo preparation**: curate 2–3 showcase queries with representative outputs
2. **Performance profiling**: identify bottlenecks in the serial report generation pipeline
3. **Model upgrade evaluation**: test with stronger LLMs for improved deliberation and report quality
4. **Frontend migration planning**: evaluate React/Vue migration for long-term maintainability
5. **Final dissertation writing**: integrate all milestone documents into the capstone thesis
