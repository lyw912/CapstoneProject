# Query Agent v2.0 — 开发总结与协作指南

> 作者：成员A | 更新：2026-04-06
> 目标读者：其他组员，了解 QueryAgent 做了什么、怎么运转、怎么对接

---

## 一、我做了什么升级

### 1.1 与原项目 QueryEngine 的对比

原项目的 `QueryEngine/agent.py` 本质是一个固定流水线脚本，不是真正的 Agent。

| 维度         | 原项目 QueryEngine          | 升级后 Query Agent v2                                                   |
| ------------ | --------------------------- | ----------------------------------------------------------------------- |
| **架构**     | 固定线性流水线，6节点串行   | LangGraph 子图，8节点含条件边和循环                                     |
| **搜索策略** | 单源 Tavily，LLM 自由选工具 | 多源并行（Tavily + Anspire + InsightDB可选），**按立场路由**            |
| **立场意识** | 无                          | 5维立场矩阵（official / support / oppose / neutral / background）       |
| **来源评估** | 无，所有来源等权            | TrustScore 4维评分（域名权威 + 时效 + 内容质量 + 搜索排名）             |
| **去重**     | 无                          | URL精确去重 + MinHash LSH内容去重                                       |
| **终止条件** | 固定 MAX_REFLECTIONS 次     | **SCS驱动的自适应终止**，立场覆盖足够才停                               |
| **输出格式** | 非结构化 Markdown 报告      | 结构化 `QueryAgentOutput`（JSON，含来源、立场分布、观点聚类、知识缺口） |

### 1.2 新增的目录结构（原有文件均未修改）

```
QueryEngine/
├── graph/                      ← 全部新增
│   ├── state.py                # LangGraph 状态定义（TypedDict）
│   ├── builder.py              # 图构建，含条件边 coverage_router
│   └── nodes/                  # 8个节点实现
│       ├── query_planner.py
│       ├── unified_search.py
│       ├── dedup_filter.py
│       ├── trust_scorer.py
│       ├── stance_classify.py
│       ├── coverage_check.py
│       ├── gap_filler.py
│       └── output_assemble.py
├── classifiers/                ← 全部新增
│   ├── trust_scorer.py         # 4维TrustScore
│   └── stance_classifier.py   # 规则+关键词混合分类
├── fusion/                     ← 全部新增
│   ├── rrf.py                  # Reciprocal Rank Fusion (SIGIR 2009)
│   └── dedup.py                # MinHash LSH去重
├── tools/
│   └── search_dispatcher.py   ← 新增，统一调度 Tavily/Anspire/InsightDB
└── evaluation/                 ← 全部新增
    ├── metrics.py              # SCS/SDI/SBS/TSM 计算
    ├── test_queries.py         # 20条标准测试集
    └── run_evaluation.py       # CLI评估脚本
```

`agent.py` 新增了 `research_structured()`、`research_structured_sync()`、`query_graph` 属性和 `_write_forum_finding()`，**原有 `research()` 方法完全保留**。

### 1.3 新增核心算法优化

#### 权威来源优先去重

- **实现文件**：[dedup_filter.py]
- **优化点**：系统现在会识别官方域名（如 `.gov.cn`, `xinhua.net` 等）。在进行 URL 和内容去重时，如果存在相似内容，会**优先保留官方或权威来源**，避免因其他站点先被爬取而导致权威信息被剔除。

---

## 二、创新点与文献支撑

### 2.1 与现有系统的对比

| 系统                 | 查询分解             | 立场感知        | 闭环终止       | 可信度评估        |
| -------------------- | -------------------- | --------------- | -------------- | ----------------- |
| GPT-Researcher       | what/why/how维度     | ❌              | ❌ 固定轮次    | ❌                |
| STORM (ACL 2024)     | 专家角色（知识背景） | ❌ 角色≠立场    | ❌             | ❌                |
| MindSearch (2024)    | DAG逻辑子问题        | ❌              | ❌             | ❌                |
| Self-RAG (ICLR 2024) | 无分解               | ❌              | 相关性自反思   | ❌                |
| **Query Agent v2**   | **5维立场矩阵**      | **✅ 核心设计** | **✅ SCS驱动** | **✅ TrustScore** |

### 2.2 创新点1：立场矩阵子查询规划

现有所有 Deep Research Agent 都在**查询生成阶段**不考虑立场多样性。本方案是唯一在查询生成阶段就注入立场约束的系统（上游多样化，而非下游重排）。

```python
# QueryEngine/graph/nodes/query_planner.py
# LLM 生成时强制覆盖5个立场维度
# 兜底：_ensure_stance_coverage() 保证即使 LLM 漏了某个立场也会补上
```

**文献依据**：Draws et al. (SIGIR 2021) — 搜索结果立场偏差测量；MMR/xQuAD（后处理多样性，我们在上游解决）

### 2.3 创新点2：闭环立场覆盖检查

Self-RAG / CRAG 的反思回路评估**信息相关性**，本方案的反思回路评估**立场覆盖度**（全新维度）。

```
SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
thresholds = {support:2, oppose:2, official:1, neutral:1}

SCS < 1.0 且 轮次 < 3  →  GapFiller 生成补搜  →  回到搜索
SCS = 1.0 或 轮次 = 3  →  强制输出
```

**文献依据**：Self-RAG (ICLR 2024), CRAG (arXiv 2401.15884), Adaptive-RAG (arXiv 2403.14403)

### 2.4 创新点3：多维可信度评分（TrustScore）

```python
# QueryEngine/classifiers/trust_scorer.py
score = 0.30 * domain_authority    # 60+ 个域名权威性字典
      + 0.25 * timeliness          # 7天半衰期指数衰减
      + 0.25 * content_quality     # snippet 长度 + 是否有全文
      + 0.20 * rrf_score           # 搜索 API 相关性得分
```

---

## 三、现在的架构

### 3.1 Query Agent 内部执行流

```
用户查询 (str)
    │
    ▼ query_planner
    │  LLM 生成 5-8 个子查询，覆盖5维立场
    │  official → 注入官方域名过滤（gov.cn, xinhua.net 等）
    │
    ▼ unified_search（asyncio.gather 并行）
    │  target_source=tavily    → TavilyNewsAgency（国际新闻）
    │  target_source=anspire   → AnspireAISearch（中文媒体）
    │  target_source=insight_db → MediaCrawlerDB（MindSpider社媒数据，可选）
    │
    ▼ dedup_filter
    │  URL精确去重 → MinHash LSH内容去重（Jaccard ≥ 0.8 视为重复）
    │
    ▼ trust_scorer
    │  每条来源计算 trust_score ∈ [0, 1]
    │
    ▼ stance_classify
    │  官方域名规则(0.90) > 关键词匹配(0.50-0.85) > 子查询弱标签(0.50) > 默认neutral(0.40)
    │
    ▼ coverage_check
    │  计算 SCS，识别 missing_stances
    │
    ├─ 覆盖足够 ────────────────────────────────┐
    ├─ 达到3轮上限 ──────────────────────────────┤
    └─ 有缺失立场 → gap_filler（LLM生成补搜）     │
                    └→ 回到 unified_search        │
                                                 ▼
                                         output_assemble
                                         立场分布 + OpinionCluster(LLM)
                                         + KnowledgeGaps(LLM)
                                         → QueryAgentOutput (dict)
```

### 3.2 QueryAgentOutput 数据结构

```python
# QueryEngine/graph/state.py — QueryAgentOutput TypedDict
{
    "original_query":     str,
    "analysis_type":      str,              # event/brand/policy/person/general
    "search_iterations":  int,              # 实际搜索轮次（1-3）
    "total_sources_found": int,             # 去重前总数
    "total_sources_kept":  int,             # 去重后保留数
    "stance_distribution": {               # 各立场占比（0-1）
        "support": 0.30, "oppose": 0.20,
        "official": 0.15, "neutral": 0.25, "background": 0.10
    },
    "opinion_clusters": [                  # 每个立场的LLM聚类
        {
            "stance":               "oppose",
            "core_argument":        "核心论点（1句话）",
            "representative_quote": "代表性原文引用",
            "source_count":         5,
            "estimated_proportion": 0.20,
        }, ...
    ],
    "sources": [                           # SourceItem列表，按trust_score降序
        {
            "url": str, "title": str, "snippet": str,
            "source_api": "tavily"/"anspire"/"insight_db",
            "trust_score": 0.73,           # 0-1
            "stance_label": "oppose",      # 立场标签
            "stance_confidence": 0.80,
            "platform": "reuters.com",
        }, ...
    ],
    "knowledge_gaps":    ["我们尚不清楚...", ...],
    "coverage_score":    0.875,            # SCS 值
    "structured_summary": "",             # Phase 3 待实现
    "trace_log":         [...],
}
```

### 3.3 与整个 BettaFish 系统的关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     BettaFish 全系统                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  信息获取层                                                │    │
│  │                                                           │    │
│  │  MindSpider ──爬取7个社媒平台──→ MySQL                    │    │
│  │      ↑ 独立后台进程                  ↑                    │    │
│  │                          InsightEngine.MediaCrawlerDB    │    │
│  │                                      ↑ 可选第三源         │    │
│  │  [Query Agent v2]──────────────────────────────────┐     │    │
│  │  Tavily（国际）+ Anspire（中文）+ InsightDB（可选）  │     │    │
│  │  → 立场矩阵规划 → 多源并行搜索 → 去重+评分+分类      │     │    │
│  │  → QueryAgentOutput（结构化JSON）                   │     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                         ↓ 文字输出写日志                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  协作层（ForumEngine）                                     │    │
│  │                                                           │    │
│  │  query.log ──┐                                           │    │
│  │  media.log ──┼→ LogMonitor → forum.log ←→ 前端SocketIO  │    │
│  │              └→ 每5条发言 → ForumHost LLM → [HOST]总结   │    │
│  │                                 ↓                         │    │
│  │              各SummaryNode 读取最新HOST发言注入Prompt      │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  MediaEngine（Bocha中文搜索）    InsightEngine（MySQL查询）        │
│  ReportEngine（Markdown报告）    Flask主应用（系统编排）            │
└─────────────────────────────────────────────────────────────────┘
```

**三引擎信息流向（互相独立，不共享搜索结果）：**

| 引擎                         | 数据来源                     | 写入        | 读取                  |
| ---------------------------- | ---------------------------- | ----------- | --------------------- |
| QueryEngine (原流程)         | Tavily                       | query.log   | forum.log（HOST发言） |
| **Query Agent v2（新流程）** | Tavily + Anspire + InsightDB | 暂未接入    | —                     |
| MediaEngine                  | Bocha                        | media.log   | forum.log             |
| InsightEngine                | MySQL                        | insight.log | forum.log             |

---

## 四、其他同学怎么对接

### 4.1 调用方式

```python
from QueryEngine.agent import DeepSearchAgent

agent = DeepSearchAgent()

# 异步调用（推荐，在 async 函数中）
output = await agent.research_structured("DeepSeek发布新模型 各方舆论")

# 同步调用（兼容非 async 环境）
output = agent.research_structured_sync("DeepSeek发布新模型 各方舆论")

# output 是 QueryAgentOutput dict，字段见上方 3.2
```

### 4.2 快速评估

```bash
# 快速验证（只跑 Q01）
python -m QueryEngine.evaluation.run_evaluation --quick

# 指定查询 ID
python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16

# 完整20条评估
python -m QueryEngine.evaluation.run_evaluation --full
```

### 4.3 可视化界面

```bash
streamlit run SingleEngineApp/query_agent_temp_app.py
```

### 4.3.1 新增可视化界面英文版本

```bash
streamlit run SingleEngineApp/query_agent_temp_app_en.py
```

四个 Tab：立场分布 / 来源列表（可按立场筛选）/ 观点聚类 / 知识缺口

### 4.4 ForumEngine 对接（⚠️ 当前断开，需要处理）

**问题**：ForumEngine 的 LogMonitor 监控的是这些模式：

```python
# ForumEngine/monitor.py
self.target_node_patterns = [
    'FirstSummaryNode',        # ← LangGraph节点不打这个
    '正在生成首次段落总结',     # ← LangGraph节点不打这个
]
```

`research_structured()` 用的是 LangGraph 节点，日志前缀是 `[QueryPlanner]`、`[OutputAssemble]` 等，**不会被 ForumEngine 捕获**。

**两种修复方案**（二选一）：

**方案A（推荐，改 agent.py）**：在 `research_structured()` 返回前调用已实现的 `_write_forum_finding(output)`，直接向 query.log 写入 `[FINDING]` 格式的发言。

```python
# QueryEngine/agent.py — research_structured() 最后几行加上：
self._write_forum_finding(output)   # 这个方法已实现，加一行调用即可
return output
```

**方案B（改 ForumEngine）**：在 `monitor.py` 的 `target_node_patterns` 中增加 LangGraph 节点的日志前缀：

```python
self.target_node_patterns += [
    '[QueryPlanner]', '[OutputAssemble]', '[CoverageCheck]'
]
```

### 4.5 ReportEngine 对接

ReportEngine 消费 Markdown。`_output_to_markdown()` 方法尚未实现，需要补充。模板见 `QUERY_AGENT_ARCHITECTURE_v2_PART2.md` §9.1。

### 4.6 InsightEngine 使用 QueryAgent 的结果

InsightEngine 可以直接接收 `QueryAgentOutput` dict 做二次分析，接口已稳定。

QueryAgent 使用 InsightEngine 的数据：通过 `MediaCrawlerDB` 即可（`search_dispatcher.py` 中已实现 `_search_insight_db()` 分支，需要 MindSpider 已爬取数据）。

---

## 五、现存问题与局限

### 5.1 架构层面的问题

| 问题                        | 严重度 | 说明                                                                                                                     |
| --------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| **ForumEngine协作链路断开** | 高     | `research_structured()` 日志格式不匹配 ForumEngine 监控，QueryAgent 在多Agent协作中是哑巴。修复成本极低（加一行调用）    |
| **三引擎各搜各的，不共享**  | 中     | MediaEngine 搜到的图片URL、InsightEngine 的社媒数据，QueryAgent 完全不知道。同一话题可能被三个引擎重复搜索，没有协调机制 |

### 5.2 MediaEngine 的多模态问题

MediaEngine 名为多模态智能体，实际上只处理了文字，图片和视频完全没有被利用：

```python
# MediaEngine/tools/search.py — BochaResponse 结构
images:      List[ImageResult]      # 图片URL → 没有任何代码读取 → 直接丢弃
modal_cards: List[ModalCardResult]  # 视频/天气/股票卡 → 没有节点消费 → 直接丢弃
webpages:    List[WebpageResult]    # ← 实际上所有节点只处理这个（文字）
```

**MediaEngine 的实质**：用 Bocha 中文搜索引擎搜索，处理文字内容，和 QueryEngine（用 Tavily 英文搜索）逻辑几乎完全相同，差异只是搜索后端。

**最低成本的改进方向**（不需要视觉模型）：解析 Bocha 的 `video` 类型 modal_card，把视频标题、播放量、发布时间以结构化文字写入 snippet，视频元数据就进入了分析流程。这对于抖音、B站等社媒话题的舆情分析有实质价值。

### 5.3 MindSpider 与 QueryAgent 的整合

当前整合方式（Spider → MySQL → InsightEngine.MediaCrawlerDB → QueryAgent 的 InsightDB 源）。Spider 是异步后台采集（小时级），QueryAgent 是同步实时查询（分钟内） **可考虑？**。

---

_文档版本：v2.0 | 2026-04-07_
