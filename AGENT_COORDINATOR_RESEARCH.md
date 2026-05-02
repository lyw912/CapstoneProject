# AgentCoordinator 设计调研报告 v3.0

> 作者：li_yewen | 日期：2026-05-02  
> 本文档记录 AgentCoordinator 的深度架构设计，核心关注：**跨源综合机制创新**、**多维度辩论式分析**、**动态反馈驱动搜索**、**细粒度平台解读**、**回声室/信息茧房破解**、**Report 呈现创新**。

---

## 目录

1. [核心设计理念](#一核心设计理念)
2. [★ 创新点总览（6 大创新）](#二创新点总览)
3. [创新 1：Multi-Perspective Deliberation Engine（多维度辩论引擎）](#三创新1多维度辩论引擎)
4. [创新 2：CRAG 驱动的动态反馈搜索（Coordinator → Agent 反向驱动）](#四创新2crag驱动的动态反馈搜索)
5. [创新 3：Platform-Aware Deep Interpretation（平台感知深度解读）](#五创新3平台感知深度解读)
6. [创新 4：Echo Chamber Breaker（回声室/信息茧房破解 + 事实分离）](#六创新4回声室信息茧房破解)
7. [创新 5：Cross-Source Divergence Matrix（跨源分歧矩阵）](#七创新5跨源分歧矩阵)
8. [创新 6：Evidence-Traced Report Architecture（证据溯源报告架构）](#八创新6证据溯源报告架构)
9. [AgentCoordinator 完整架构设计](#九agentcoordinator完整架构设计)
10. [LangGraph 实现设计](#十langgraph实现设计)
11. [容错与降级设计](#十一容错与降级设计)
12. [实现路径与优先级](#十二实现路径与优先级)
13. [学术支撑汇总](#十三学术支撑汇总)

---

## 一、核心设计理念

### 1.1 AgentCoordinator 不是胶水层

**常见误区**：Coordinator = 并行启动两个 Agent → 拼接结果 → 交给 Report。

**我们的理解**：Coordinator 是**整个系统最核心的智能层**，它需要：

1. **模拟真实舆论场**：每个维度/视角不是孤立分析，而是听到其他维度的发言后，重新考虑自己的立场
2. **动态驱动搜索**：不只是"拿到结果再处理"，而是发现信息不足时**反向驱动 Agent 补搜**
3. **细粒度解构**：不是简单聚合"23条帖子"，而是识别微博大众情绪化 vs 知乎理性分析 vs B站年轻多元背后的**社会结构差异**
4. **穿透回声室**：主动识别舆论中的从众效应、沉默的大多数、信息茧房，用事实和多维度框架重新审视
5. **可溯源呈现**：最终报告中的每个结论都能追溯到具体数据源、分析路径、和辩论过程

### 1.2 与原 ForumEngine 的本质区别

| 维度 | 原 ForumEngine | 新 AgentCoordinator |
|------|---------------|-------------------|
| 协调方式 | 文件总线 + 30s 轮询 | LangGraph 状态机 + 内存直传 |
| 交互深度 | LLM 主持人读日志写建议 | **结构化辩论引擎**：多轮质证 + 命题级交叉验证 |
| 搜索能力 | 无反馈搜索 | **CRAG 反向驱动**：发现信息缺口 → 指导 Agent 补搜 |
| 分析粒度 | 平台无差别聚合 | **平台画像 + 人群分层** |
| 偏见修正 | 无 | **回声室检测 + 沉默大多数补偿 + 事实分离层** |
| 呈现方式 | 纯文本报告 | **证据溯源 + 可视化分歧矩阵 + 辩论轨迹** |

---

## 二、★ 创新点总览

| # | 创新点 | 核心价值 | 学术支撑 |
|---|--------|---------|---------|
| 1 | **Multi-Perspective Deliberation Engine** | 模拟真实舆论场的多维度辩论，每个维度倾听并回应其他维度 | AI Council（arXiv:2604.26561），MAD（EMNLP 2024），MPSR（MDPI 2025） |
| 2 | **CRAG 驱动的动态反馈搜索** | Coordinator 发现信息缺口后反向驱动 Query/MindSpider 补搜 | CRAG（arXiv:2401.15884），FAIR-RAG（arXiv:2510.22344），Agentic RAG |
| 3 | **Platform-Aware Deep Interpretation** | 不同社媒平台的人群画像 × 立场差异 = 社会结构性洞察 | 跨平台情感分析（2024 综述），中国社媒人群差异研究 |
| 4 | **Echo Chamber Breaker** | 识别回声室效应 + 补偿沉默大多数 + 将舆论与事实分离 | COLING 2025 回声室检测，LLM Filter Bubble（Northeastern 2025） |
| 5 | **Cross-Source Divergence Matrix** | 不只计算一个 δ 值，而是生成多维分歧矩阵（按立场×来源×平台） | CSSD 扩展，KL 散度（PLOS ONE 2025），MoA（ICLR 2025） |
| 6 | **Evidence-Traced Report Architecture** | 报告中每个结论可追溯到数据源 + 辩论过程 + 分歧/共识标注 | Anthropic Artifact System，CRAG 证据链 |

---

## 三、创新 1：Multi-Perspective Deliberation Engine（多维度辩论引擎）

### 3.1 你的设想的可行性分析

> 用户设想："每个维度发言，提出论据；另一个维度支持、反驳；第一个维度倾听后再发言。"

**这个模式在学术上被称为 Multi-Agent Deliberation（多智能体审议）**，有大量前沿支撑：

**支持证据**：
- **AI Council**（arXiv:2604.26561，2025）：专门设计了"保留分歧"的三阶段审议系统，120 场审议实验证明架构异构性（不同 LLM 代表不同视角）**显著减少人工共识**，Cohen's d 效应量大
- **MAD**（Du et al.，EMNLP 2024）：多 Agent 迭代辩论改善推理质量
- **RADAR**（arXiv:2604.19005）：Politician Agent vs Scientist Agent vs Judge 三角结构

**需要修正的部分**：
- NeurIPS 2025 poster 发现：**纯 debate 的期望正确率不高于投票**。关键在于需要"biased belief update"——即每个 Agent 只在收到**高置信度、有新证据的反驳**时才修改立场
- Google DeepMind 2025 警告：无结构的多 Agent 网络错误放大 17.2×。必须用**结构化拓扑**

### 3.2 设计方案：Structured Deliberation Protocol

```
Phase 1: Independent Analysis（独立分析）
  每个维度（Perspective Agent）基于 Query/Media Agent 的原始数据
  独立产出一份"维度分析报告"
  报告包含：核心论点 + 支撑证据 + 置信度

Phase 2: Cross-Examination（交叉质证）——你的设想的工程化实现
  每个维度轮流：
    1. 读取所有其他维度的 Phase 1 报告
    2. 对每个他方论点给出：
       - AGREE（赞同，引用自己的数据作为印证）
       - CHALLENGE（质疑，提出反例或指出证据不足）
       - SUPPLEMENT（补充，给出对方遗漏的信息）
    3. 基于他方的质疑和补充，修订自己的分析（附理由）

Phase 3: Synthesis（综合裁定）
  Synthesis Agent（高质量 LLM）：
    - 汇总所有维度的最终立场（含修订后版本）
    - 标注哪些是"跨维度共识"，哪些是"持续分歧"
    - 对持续分歧给出"为什么分歧"的分析
    - 输出结构化的 SynthesisReport
```

### 3.3 维度（Perspective）是什么——动态而非固定

**关键设计**：维度不是固定的 5 种立场，而是**根据议题类型动态生成**。

```python
PERSPECTIVE_TEMPLATES = {
    "event": [                          # 突发事件
        "事件经过与事实核查维度",
        "社会影响与情感共鸣维度",
        "政策/法律/制度反思维度",
        "历史类比与趋势判断维度",
    ],
    "brand": [                          # 品牌舆情
        "消费者体验与情感维度",
        "企业战略与行业竞争维度",
        "媒体报道框架与叙事维度",
        "投资者/市场反应维度",
    ],
    "policy": [                         # 政策讨论
        "政策制定者视角",
        "受影响群体视角",
        "经济/技术可行性视角",
        "国际比较与历史经验视角",
    ],
    "technology": [                     # 技术话题
        "技术原理与工程事实维度",
        "产业应用与商业影响维度",
        "公众认知与情感反应维度",
        "伦理/安全/社会影响维度",
    ],
    "general": [                        # 通用
        "事实与数据维度",
        "公众情感与社会心理维度",
        "利益相关者分析维度",
        "历史/哲学/跨领域反思维度",
    ],
}
```

QueryAgent 的 `analysis_type`（event/brand/policy/person/general）直接驱动维度选择。

### 3.4 为什么不是纯 Prompt？——工程化实现

**纯 Prompt 方案**（一次性让 LLM 假装多个角色辩论）的问题：
- 同一 LLM 的"内部辩论"会产生**从众偏差**（conformity bias），所有"角色"收敛到同一结论
- AI Council 论文实证：**prompt engineering alone cannot achieve** 真正的观点保留

**工程化方案**：

```
方案 A：Multi-Call Separation（多次调用分离）
  - 每个维度是一次独立的 LLM 调用
  - 调用之间传递结构化的 agree/challenge/supplement 数据
  - 优点：真正的上下文隔离，防止从众
  - 缺点：LLM 调用次数 = 维度数 × 轮数（4×2=8 次）

方案 B：Structured Prompt with Role Anchoring（结构化 Prompt + 角色锚定）
  - 单次调用，但 Prompt 中明确要求：
    "对于每个维度，你必须先独立分析，再显式标注是否同意/反驳其他维度"
    "如果所有维度都同意某观点，你必须主动提出一个反面论证"
  - 优点：单次调用，快速
  - 缺点：从众风险，但可以通过 "Devil's Advocate" 约束缓解

方案 C（推荐）：Hybrid — Phase 1 多调用 + Phase 2-3 单调用
  - Phase 1：4 个维度 × 1 次调用 = 4 次独立调用（确保真正独立）
  - Phase 2：1 次调用，输入所有 Phase 1 结果，要求交叉质证
  - Phase 3：1 次调用，综合裁定
  - 总计：6 次 LLM 调用，平衡质量和效率
```

### 3.5 关键约束（防止"跑题"）

MAD 论文（arXiv:2502.19559）指出的 "problem drift"（问题漂移）风险——辩论过程中 Agent 偏离原始问题。

**解决方案**：每次 LLM 调用的 Prompt 都**强制以原始查询和原始数据为锚点**，不允许引入 LLM 内部知识作为事实依据。所有论据必须引用 QueryAgent/MediaAgent 的具体数据源。

---

## 四、创新 2：CRAG 驱动的动态反馈搜索（Coordinator → Agent 反向驱动）

### 4.1 核心问题

当前系统是**单向流动**：Agent → Coordinator → Report。如果 Coordinator 发现信息不足怎么办？

### 4.2 学术支撑

**CRAG**（Corrective RAG，arXiv:2401.15884，ICLR 2024 Workshop）：

```
检索 → 评估检索质量 → 三路决策：
  CORRECT：直接使用
  AMBIGUOUS：结合内部检索 + 外部 Web 搜索
  INCORRECT：放弃内部检索，转 Web 搜索
```

**FAIR-RAG**（arXiv:2510.22344）：迭代式检索精炼循环：
```
检索 → 结构化证据评估（SEA）→ 证据不足？→ 精炼查询 → 再次检索
```

**Agentic RAG** 的核心理念：检索不是一次性的，而是一个反馈循环。

### 4.3 在我们系统中的实现

```
Coordinator 的 Deliberation Engine 输出 → Gap Detector（缺口检测）
  ↓
  检测到以下缺口：
    a) 某个维度的论据数量 < 阈值（例如"经济影响"维度只有 1 条来源）
    b) 某立场的来源只覆盖 1 个平台（例如"反对"只有微博数据）
    c) 分歧点缺乏事实验证（例如"是否涨价"是核心分歧但没有价格数据）
  ↓
  生成 Targeted Search Directives（定向搜索指令）：
    [
      {"query": "xxx 价格变化 实际数据", "source": "tavily", "rationale": "验证涨价争议"},
      {"query": "xxx", "source": "mindspider_db", "rationale": "补充知乎理性分析"},
    ]
  ↓
  轻量级补搜（Query Agent 的 unified_search 节点单独调用，非重启整个 Agent）
  ↓
  补搜结果注入 Deliberation Engine 的下一轮分析
```

**三种补搜通道**（按延迟从低到高）：

| 通道 | 延迟 | 适用场景 | 实现方式 |
|------|------|---------|---------|
| **MindSpider DB 查询** | ~100ms | 补充特定平台数据 | `MindSpiderDB.search_topic_globally(keyword)` |
| **Tavily/Anspire API 搜索** | ~2-5s | 补充 Web 搜索事实数据 | `search_dispatcher.dispatch([SubQueryItem])` |
| **MindSpider BroadTopicExtraction** | ~30s | 数据完全空白时触发后台话题提取 | `subprocess.Popen`（不阻塞，已有实现） |

**注意**：DeepSentimentCrawling（Playwright 深度爬取）**不适合实时场景**（耗时分钟级），仅在后台定时任务中运行。

### 4.4 LangGraph 中的实现——条件回边

```python
# LangGraph 条件边实现反馈循环
graph.add_conditional_edges(
    "deliberation_engine",
    gap_detector,                    # 缺口检测函数
    {
        "sufficient": "synthesis",   # 信息充足 → 直接综合
        "need_search": "targeted_search",  # 需要补搜
        "max_rounds": "synthesis",   # 达到最大轮次 → 强制综合
    },
)
graph.add_edge("targeted_search", "deliberation_engine")  # 补搜后回到辩论
```

**安全机制**：最多 1 轮补搜（防止无限循环），总超时 120s。

---

## 五、创新 3：Platform-Aware Deep Interpretation（平台感知深度解读）

### 5.1 核心认识

Query Agent Phase 3 已经实现了 `per_platform` 立场分布统计（微博 50% 支持 vs 知乎 87.5% 中立 vs B站 40% 支持）。但这只是**数字**，没有**解读**。

**真正的价值**：为什么不同平台会有不同立场？这不是随机的，而是反映了**用户人群结构差异**。

### 5.2 中国六大社媒平台画像（来自调研数据）

| 平台 | 典型用户 | 内容特征 | 讨论风格 | 对舆情分析的意义 |
|------|---------|---------|---------|---------------|
| **微博** | 大众化，20-40岁，城市 | 短文本、情绪化、热搜驱动 | 情绪宣泄、站队、快速传播 | 反映**即时情绪反应**，易被水军/营销号影响 |
| **知乎** | 高学历（71.5% 本科+），25-45岁 | 长文分析、引用数据 | 理性讨论、多角度论证 | 反映**知识精英立场**，可能脱离大众 |
| **B站** | Z世代（78.67% 90后/00后），22.8岁均值 | 弹幕、视频评论 | 年轻、多元、亚文化 | 反映**年轻一代态度**，可能代表未来趋势 |
| **抖音** | 18-35岁，城市为主 | 短视频、直觉驱动 | 碎片化、算法茧房严重 | 反映**大众直觉感受**，但受推荐算法严重扭曲 |
| **小红书** | 70% 女性，18-35岁，一二线城市 | 种草/测评，消费导向 | 体验分享、视觉叙事 | 反映**消费者视角**，尤其是女性群体 |
| **贴吧** | 分散，兴趣驱动 | 匿名/半匿名讨论 | 激烈、极端化、圈层固化 | 反映**特定兴趣群体的深层态度**，但代表性有限 |

### 5.3 工程化实现——Platform Interpreter Node

```python
PLATFORM_PROFILES = {
    "weibo": {
        "user_base": "大众化，20-40岁城市用户",
        "content_style": "短文本、情绪驱动、热搜导向",
        "bias_tendency": "情绪放大，水军/营销号干扰高",
        "weight_factor": 0.8,  # 因水军风险降权
        "interpretation_prompt": "微博上的 {stance} 立场可能反映大众即时情绪反应。"
                                 "需考虑：(1)是否有水军/营销号推动 (2)热搜算法放大效应 "
                                 "(3)短文本中的情绪宣泄而非深思熟虑",
    },
    "zhihu": {
        "user_base": "高学历（71.5%本科+），25-45岁知识群体",
        "content_style": "长文分析、引用数据、多角度论证",
        "bias_tendency": "精英偏见，可能脱离大众实际感受",
        "weight_factor": 1.0,
        "interpretation_prompt": "知乎上的 {stance} 立场代表高学历群体的理性分析。"
                                 "需考虑：(1)是否存在知识精英与大众的认知鸿沟 "
                                 "(2)长文讨论是否过度理想化",
    },
    "bilibili": {
        "user_base": "Z世代（78.67% 90后/00后），平均22.8岁，城市年轻人",
        "content_style": "弹幕文化、视频评论、亚文化",
        "bias_tendency": "年轻视角偏见，可能忽视中老年群体关切",
        "weight_factor": 0.9,
        "interpretation_prompt": "B站上的 {stance} 立场反映Z世代的态度，可能预示未来趋势方向。"
                                 "需考虑：(1)年龄结构极度偏年轻 (2)亚文化圈层效应 "
                                 "(3)弹幕的从众/玩梗特征",
    },
    # ... douyin, xiaohongshu, tieba
}
```

**不是简单的权重调整**，而是让 Deliberation Engine 的每个维度在讨论时**意识到数据来源的人群结构**：

> "知乎用户 87.5% 持中立态度，但需注意知乎用户群体 71.5% 为本科以上学历，这代表的是知识精英的理性审视而非大众的冷漠。同一话题在微博（大众用户）的反应截然不同（50% 支持），可能反映了不同社会阶层对此事的不同利益关切。"

---

## 六、创新 4：Echo Chamber Breaker（回声室/信息茧房破解 + 事实分离）

### 6.1 问题定义

舆论天生具有三大偏见：
1. **回声室（Echo Chamber）**：相似观点的人互相强化，形成"全世界都这样想"的假象
2. **信息茧房（Filter Bubble）**：平台推荐算法只推你爱看的，极化立场
3. **沉默的大多数（Spiral of Silence）**：温和/中间立场者不愿发声，导致极端声音被放大

### 6.2 学术支撑

- **COLING 2025**：LLM-Powered Simulations Revealing Polarization in Social Networks——用 LLM 模拟检测回声室极化
- **AI Council**（arXiv:2604.26561）：架构异构性防止人工共识，**保留合法分歧**
- **Northeastern 2025 研究**：ChatGPT 本身也会产生 Filter Bubble，需要主动干预

### 6.3 工程化实现：三层破解

```
Layer 1: Echo Chamber Detection（回声室检测）
  已有基础：Query Agent Phase 3.6 的"内容多样性检测"（diversity < 0.7 → 水军告警）
  扩展：
    - 计算 Stance Entropy = -Σ p(s) log p(s)
    - 低 Entropy + 高帖子数 = 回声室信号
    - 单一平台内 Stance Entropy < 阈值 → 标注"该平台可能存在回声室效应"

Layer 2: Silent Majority Compensation（沉默大多数补偿）
  核心逻辑：
    - 如果所有社媒平台的立场都偏向某一方（如 80%+ 支持）
    - 而 Web 搜索中发现有组织化的反对论点但社媒上几乎看不到
    - → LLM 生成 "Silent Majority Hypothesis"：
      "社媒数据显示压倒性的支持态度，但这可能受回声室效应影响。
       以下迹象表明存在沉默的反对声音：
       - Web 搜索发现了 N 篇持反对立场的深度报道
       - 社媒评论中 M 条以委婉方式表达了担忧
       - 历史上类似事件的最终走向与初期舆论不同"

Layer 3: Fact-Opinion Separation（事实-舆论分离）★ 核心创新
  目标：从众多情绪和舆论中抽离，冷静客观地分析
  
  实现：在 Synthesis 阶段，明确要求 LLM 区分两类内容：
  
  A. 可验证事实（Verifiable Facts）：
     - 数字、日期、官方声明、技术参数
     - 要求引用具体来源 URL
     - 标注置信度（单源 / 多源交叉验证）
  
  B. 观点/情感/态度（Opinions & Sentiments）：
     - 哪些人持何种观点
     - 情感强度和分布
     - 明确标注"这是观点/情感反应，不是事实"
  
  C. 深层分析框架（Analytical Frameworks）：
     - 经济学视角：成本收益分析、市场影响
     - 技术视角：可行性评估、技术路线对比
     - 历史视角：类似事件的发展规律
     - 社会学视角：利益相关者分析、社会结构因素
     - 这些由 LLM 基于事实层生成，明确标注"这是分析推断"
```

### 6.4 Fact-Opinion 分离的 Prompt 设计（工程化）

```python
FACT_OPINION_SEPARATION_PROMPT = """
You are a critical analyst. Given the following multi-source analysis data,
perform Fact-Opinion Separation:

QUERY: {query}
QUERY AGENT DATA: {query_agent_summary}
MEDIA AGENT DATA: {media_agent_summary}
SOCIAL MEDIA DATA: {social_sentiment_summary}

OUTPUT FORMAT (JSON):

{
  "verified_facts": [
    {
      "fact": "...",
      "sources": ["url1", "url2"],
      "verification_status": "cross_verified" | "single_source" | "disputed",
      "confidence": 0.0-1.0
    }
  ],
  "opinions_and_sentiments": [
    {
      "perspective": "...",
      "holders": "描述持此观点的群体特征",
      "sentiment_intensity": "strong" | "moderate" | "mild",
      "platform_distribution": {"weibo": 0.5, "zhihu": 0.1},
      "potential_biases": ["回声室", "水军", "算法推荐"]
    }
  ],
  "analytical_frameworks": [
    {
      "framework": "economic" | "technical" | "historical" | "sociological",
      "analysis": "...",
      "basis": "基于以上哪些事实得出",
      "certainty": "high" | "medium" | "speculative"
    }
  ],
  "echo_chamber_warnings": [
    "某平台观点高度一致（Entropy=X），可能存在回声室效应",
    "沉默大多数：Web搜索显示存在N篇未在社媒广泛讨论的反面报道"
  ]
}
"""
```

---

## 七、创新 5：Cross-Source Divergence Matrix（跨源分歧矩阵）

### 7.1 从单一 δ 值到多维矩阵

v2.0 只计算了一个跨 Agent CSSD 值 δ。但这太粗糙——**分歧可能存在于多个维度上**。

### 7.2 分歧矩阵设计

```
              QueryAgent  MediaAgent  微博    知乎    B站    抖音
QueryAgent      —          δ_qm      δ_qw    δ_qz   δ_qb   δ_qd
MediaAgent    δ_qm          —        δ_mw    δ_mz   δ_mb   δ_md
微博           δ_qw        δ_mw       —      δ_wz   δ_wb   δ_wd
知乎           δ_qz        δ_mz      δ_wz     —     δ_zb   δ_zd
B站            δ_qb        δ_mb      δ_wb    δ_zb    —     δ_bd
抖音           δ_qd        δ_md      δ_wd    δ_zd   δ_bd    —
```

每个 δ 值 = `1 - cosine_sim(stance_vector_A, stance_vector_B)`

**可视化**：在报告中生成热力图，颜色越深 = 分歧越大。

### 7.3 分歧矩阵的解读价值

- **Web 搜索 vs 社媒全体**：反映"官方叙事 vs 民众反应"差距
- **微博 vs 知乎**：反映"情绪化反应 vs 理性分析"差距
- **B站 vs 其他**：反映"年轻一代 vs 主流"差距
- **某个来源 vs 所有其他**：如果某个来源与所有其他都分歧大，可能是**该来源被污染**（水军/算法推荐）

---

## 八、创新 6：Evidence-Traced Report Architecture（证据溯源报告架构）

### 8.1 目标

报告不是"AI 写了一篇作文"，而是**像学术论文一样有据可查、结构严谨的分析报告**。

核心原则：
- **数据与分析分离**：第 4 章呈现客观数据和一手证据（占报告主体），第 7 章基于数据给出克制的分析结论
- **所有论据可溯源**：社媒帖子附可点击跳转 URL，网络来源附 TrustScore 和立场标注
- **指标先定义后使用**：CSSD、SCS、TrustScore、Shannon 熵等自定义指标在方法论章节给出公式和阈值表
- **Phase 3 可视化预留**：热力图、辩论交互、置信度仪表盘在 Markdown 中标注 `[PLACEHOLDER]`，由 ReportAgent 渲染

### 8.2 报告结构设计（IMRaD 学术论文模式）

```
# 多源舆情深度分析报告

■ 摘要 (Abstract)
  - 主旨段：概括核心发现
  - 关键数字：N条来源，M个平台，S条帖子，T条评论
  - 置信度评级：★★★★☆ (78%)

■ 1. 引言与背景 (Introduction & Background)
  - 方法论创新概述（5 点）

■ 2. 方法论与指标定义 (Methodology & Metrics)
  ├── 2.1 数据采集架构（三层表格）
  ├── 2.2 核心分析指标（CSSD 公式 + 阈值表、SCS、TrustScore 公式、Shannon 熵）
  └── 2.3 辩论引擎方法（Hybrid Plan C + 维度选择 + 三阶段流程）

■ 3. 数据概览 (Data Overview)
  ├── 3.1 网络搜索来源（总数、SCS 值、立场分布表）
  └── 3.2 社交媒体数据（帖子数/评论数、CSSD 值、各平台分布对比表）

■ 4. 研究发现 (Findings) —— 占报告主体，客观数据 + 一手证据
  ├── 4.1 可验证事实（✅/⚠️ 验证状态 + 置信度 + 来源引用）
  ├── 4.2 跨源分歧分析（CSSD 矩阵 + 热点解读 + [热力图 PLACEHOLDER]）
  ├── 4.3 代表性原声与证据
  │   ├── 4.3.1 网络媒体代表来源（按立场分组，附 URL + TrustScore）
  │   └── 4.3.2 社交媒体代表帖子（原文引用 + 🔗跳转链接）
  └── 4.4 评论区情感分析（按点赞数排序的高赞热评 + 立场分布）

■ 5. 多维度辩论分析 (Multi-Perspective Deliberation)
  ├── 5.1 第一阶段：独立分析
  ├── 5.2 第二阶段：交叉质证（初步共识 + 质证分歧）
  ├── 5.3 第三阶段：综合裁定
  ├── 5.4 跨维度共识（去重后列表）
  ├── 5.5 持续分歧（认知边界声明）
  └── [辩论交互 PLACEHOLDER]

■ 6. 偏见评估与信息完整性 (Bias Assessment)
  ├── 6.1 回声室检测（Shannon 熵检查）
  └── 6.2 沉默的大多数假设

■ 7. 结论与启示 (Conclusions & Implications) —— 克制、客观
  ├── 7.1 平台差异的社会学解读（各平台画像 + 人群解读）
  ├── 7.2 分析框架（🟢/🟡/🔴 确定性等级）
  ├── 7.3 核心矛盾与张力
  ├── 7.4 局限性声明（5 条）
  └── 7.5 建议进一步研究方向
  └── [置信度仪表盘 PLACEHOLDER]

■ 附录 (Appendices)
  ├── A. 完整来源列表（<details> 可折叠）
  ├── B. 跨源分歧矩阵原始数据（<details> 可折叠）
  ├── C. 分析流程追踪日志（<details> 可折叠）
  ├── D. 分歧矩阵热力图 [VISUALIZATION PLACEHOLDER]
  ├── E. 辩论过程交互式时间线 [INTERACTIVE PLACEHOLDER]
  └── F. Flask WebSocket 实时进度 [FEATURE PLACEHOLDER]
```

### 8.3 实现文件

| 文件 | 职责 |
|------|------|
| `AgentCoordinator/academic_report_generator.py` | **核心生成器**：接收 `coordinator_output.json` 格式 dict，输出完整学术风格 Markdown |
| `AgentCoordinator/graph/nodes/report_agent_node.py` | LangGraph 节点：从 CoordinatorState 组装生成器输入，调用 `generate_academic_report()` |
| `AgentCoordinator/coordinator_output_schema.py` | JSON Schema 定义 + `build_coordinator_output()` 标准化构建 |

**关键设计**：
- `report_agent_node.py` 中 `_build_generator_input()` 从 LangGraph state 中提取全部字段，组装为与 `coordinator_output.json` 相同格式的 dict
- `academic_report_generator.py` 中 `generate_academic_report()` 为纯函数（无 LLM 调用、无 I/O），只做数据 → Markdown 的确定性转换
- 每次 `coordinator.run()` 都会自动生成此格式的报告，不是一次性手动编辑

### 8.4 呈现创新点（如何在报告中体现我们的工作）

| 报告元素 | 体现的创新 | 直观效果 |
|---------|----------|---------|
| 事实层 vs 舆论层分离 | Echo Chamber Breaker | 读者一眼区分"发生了什么"vs"大家怎么看" |
| 分歧矩阵热力图 | Cross-Source Divergence Matrix | 可视化呈现多源分歧，比单一 δ 值直观 100 倍 |
| 多维度辩论轨迹 | Deliberation Engine | 展示"AI 如何从多角度深思"的过程，不是黑盒 |
| 平台画像标签 | Platform-Aware Interpretation | "微博（大众情绪）56% 支持 vs 知乎（知识群体）87% 中立" |
| 偏见声明 | Echo Chamber + Silent Majority | 学术诚实性，体现系统自我反思能力 |
| 置信度标注 | Fact-Opinion Separation | 每个结论标注"多源验证"/"单源"/"推测性" |

---

## 九、AgentCoordinator 完整架构设计

### 9.1 完整数据流

```
用户查询 "DeepSeek发布新模型 各方舆论"
    ↓
[AgentCoordinator.run(query)]
    │
    ├── Phase 0: 并行执行 Agent ─────────────────────────────────────
    │   ├── [query_agent_node] → QueryAgentOutput（结构化）
    │   │     含：stance_distribution, opinion_clusters,
    │   │         social_sentiment (CSSD, per_platform, trends), sources
    │   └── [media_agent_node] → MediaReport（Markdown）
    │         含：多媒体报道分析，图片/视频引用
    │
    ├── Phase 1: 数据桥接 + 分歧矩阵 ─────────────────────────────
    │   ├── [data_bridge_node]
    │   │     - QueryAgentOutput → 结构化命题列表
    │   │     - MediaReport → 提取关键声明列表
    │   │     - 统一为 BridgedProposition 格式
    │   │
    │   └── [divergence_matrix_node]
    │         - 计算跨源分歧矩阵（Agent×Agent + Platform×Platform）
    │         - 标注分歧热点区域
    │
    ├── Phase 2: 多维度辩论 ─────────────────────────────────────────
    │   ├── [perspective_generator_node]
    │   │     根据 analysis_type 选择 4 个维度
    │   │
    │   ├── [deliberation_engine_node]  ← ★ 核心创新节点
    │   │     Phase 2.1：4 个维度独立分析（4 × LLM 调用）
    │   │     Phase 2.2：交叉质证（1 × LLM 调用）
    │   │     Phase 2.3：综合裁定（1 × LLM 调用）
    │   │
    │   └── [gap_detector_node] → conditional edge
    │         - 信息充足 → Phase 3
    │         - 信息不足 → [targeted_search_node] → 回到 Phase 2
    │
    ├── Phase 3: 回声室破解 + 事实分离 ──────────────────────────────
    │   ├── [echo_chamber_detector_node]
    │   │     - Stance Entropy 计算
    │   │     - 沉默大多数检测
    │   │     - 输出 echo_warnings
    │   │
    │   └── [fact_opinion_separator_node]
    │         - 事实层提取（可验证、有来源、交叉验证）
    │         - 观点层整理（谁持什么观点、情感强度）
    │         - 分析框架层（经济/技术/历史/社会学视角）
    │
    ├── Phase 4: 综合 + 报告生成 ───────────────────────────────────
    │   ├── [platform_interpreter_node]
    │   │     - 为每个平台数据加注人群画像解读
    │   │
    │   ├── [synthesis_node]
    │   │     - MoA 风格聚合（不是选择，而是真正综合）
    │   │     - 输出 SynthesisContext（给 Report Agent 用）
    │   │
    │   └── [report_agent_node]
    │         - 调用 ReportAgent.generate()
    │         - 传入：事实层 + 舆论层 + 辩论轨迹 + 分歧矩阵 + 偏见声明
    │
    └── 输出：CoordinatorState（含 report_html + trace_log）
```

### 9.2 CoordinatorState 完整设计

```python
class CoordinatorState(TypedDict):
    # 输入
    query: str
    analysis_type: str                         # event/brand/policy/technology/general

    # Phase 0：Agent 执行结果
    query_run: Optional[AgentRunResult]
    media_run: Optional[AgentRunResult]
    agent_errors: Annotated[List[str], operator.add]

    # Phase 1：桥接 + 分歧
    bridged_propositions: Optional[List[BridgedProposition]]
    divergence_matrix: Optional[Dict]          # {(source_a, source_b): delta_value}
    divergence_hotspots: Optional[List[str]]   # 分歧热点描述

    # Phase 2：辩论
    perspectives: Optional[List[str]]          # 本次使用的维度
    deliberation_rounds: Optional[List[Dict]]  # 辩论过程记录
    deliberation_consensus: Optional[List[str]]  # 共识点
    deliberation_dissents: Optional[List[str]]   # 持续分歧点

    # Phase 2.5：补搜
    search_gaps: Optional[List[Dict]]          # 检测到的信息缺口
    supplementary_results: Optional[List[Dict]]  # 补搜获得的新数据
    search_rounds: int                         # 补搜轮次（最多 1）

    # Phase 3：偏见修正 + 事实分离
    echo_warnings: Optional[List[str]]
    silent_majority_hypothesis: Optional[str]
    verified_facts: Optional[List[Dict]]
    opinions_sentiments: Optional[List[Dict]]
    analytical_frameworks: Optional[List[Dict]]

    # Phase 4：综合 + 报告
    platform_interpretations: Optional[Dict]   # {platform: interpretation_text}
    synthesis_context: Optional[Dict]          # 传给 Report Agent 的完整上下文
    synthesis_confidence: float
    report_output: Optional[str]               # HTML 报告

    # 全程追踪
    coordinator_trace: Annotated[List[str], operator.add]
```

---

## 十、LangGraph 实现设计

### 10.1 图结构

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(CoordinatorState)

# Phase 0：并行执行 Agent
graph.add_node("query_agent", query_agent_node)
graph.add_node("media_agent", media_agent_node)
graph.add_edge(START, "query_agent")
graph.add_edge(START, "media_agent")

# Phase 1：桥接 + 分歧矩阵（Fan-in，等待两者完成）
graph.add_node("data_bridge", data_bridge_node)
graph.add_node("divergence_matrix", divergence_matrix_node)
graph.add_edge("query_agent", "data_bridge")
graph.add_edge("media_agent", "data_bridge")
graph.add_edge("data_bridge", "divergence_matrix")

# Phase 2：辩论引擎（含条件回边）
graph.add_node("perspective_gen", perspective_generator_node)
graph.add_node("deliberation", deliberation_engine_node)
graph.add_node("targeted_search", targeted_search_node)
graph.add_edge("divergence_matrix", "perspective_gen")
graph.add_edge("perspective_gen", "deliberation")
graph.add_conditional_edges(
    "deliberation",
    gap_detector,
    {"sufficient": "echo_chamber", "need_search": "targeted_search", "max_rounds": "echo_chamber"},
)
graph.add_edge("targeted_search", "deliberation")  # 回边

# Phase 3：偏见修正 + 事实分离
graph.add_node("echo_chamber", echo_chamber_detector_node)
graph.add_node("fact_opinion", fact_opinion_separator_node)
graph.add_edge("echo_chamber", "fact_opinion")

# Phase 4：综合 + 报告
graph.add_node("platform_interpret", platform_interpreter_node)
graph.add_node("synthesis", synthesis_node)
graph.add_node("report_agent", report_agent_node)
graph.add_edge("fact_opinion", "platform_interpret")
graph.add_edge("platform_interpret", "synthesis")
graph.add_edge("synthesis", "report_agent")
graph.add_edge("report_agent", END)
```

### 10.2 目录结构

```
AgentCoordinator/
├── __init__.py
├── coordinator.py                      # 统一入口（含 checkpointing + coordinator_output.json 导出）
├── coordinator_output_schema.py        # ★ coordinator_output.json 完整 Schema 定义
├── academic_report_generator.py        # ★ 学术风格 Markdown 报告生成器
├── graph/
│   ├── builder.py                      # LangGraph 图构建（含 MemorySaver checkpointing）
│   ├── state.py                        # CoordinatorState
│   └── nodes/
│       ├── __init__.py
│       ├── query_agent_node.py         # 调用 QueryEngine（含超时 + 结果缓存）
│       ├── media_agent_node.py         # 调用 MediaEngine（含超时 + 测试数据注入）
│       ├── data_bridge_node.py         # 异构数据桥接
│       ├── divergence_matrix_node.py   # ★ 跨源分歧矩阵
│       ├── perspective_generator.py    # 动态维度生成
│       ├── deliberation_engine.py      # ★ 多维度辩论引擎
│       ├── gap_detector.py             # 信息缺口检测 + 条件路由
│       ├── targeted_search_node.py     # ★ CRAG 风格定向补搜
│       ├── echo_chamber_detector.py    # ★ 回声室检测 + 沉默大多数
│       ├── fact_opinion_separator.py   # ★ 事实-舆论分离
│       ├── platform_interpreter.py     # ★ 平台画像深度解读
│       ├── synthesis_node.py           # MoA 风格综合
│       └── report_agent_node.py        # ★ 学术风格报告生成（调用 academic_report_generator）
├── utils/
│   ├── timeout_guard.py                # asyncio 超时封装
│   ├── platform_profiles.py            # 平台画像常量
│   ├── perspective_templates.py        # 维度模板
│   └── report_bridge.py               # QueryAgentOutput → Report 格式（备用）
├── prompts/
│   ├── deliberation_prompts.py         # 辩论各阶段 Prompt
│   ├── fact_separation_prompt.py       # 事实-舆论分离 Prompt
│   └── synthesis_prompt.py             # MoA 综合 Prompt
├── cache/                              # 运行时缓存目录
│   ├── query_agent_{hash}.json         # QueryAgent 结果缓存
│   ├── coordinator_output_latest.json  # ★ 交给 ReportAgent 的结构化 JSON（每次覆盖）
│   ├── coordinator_output_{ts}.json    # 归档
│   └── test_report_{ts}.md            # 生成的 Markdown 报告
└── test_phase1.py                      # 端到端集成测试
```

---

## 十一、容错与降级设计

### 11.1 降级矩阵

| 故障节点 | 降级方案 | 对最终报告的影响 |
|---------|---------|---------------|
| Query Agent 失败 | 仅用 Media Agent 数据 | 报告标注"Web搜索数据缺失"，无 CSSD |
| Media Agent 失败 | 仅用 Query Agent 数据 | 报告标注"多媒体报道缺失" |
| 两者都失败 | 返回错误 | 无法生成报告 |
| data_bridge 失败 | 跳过命题对齐，直传原始数据 | 辩论质量降低但不中断 |
| deliberation 失败 | 跳过辩论，直接综合 | 丢失多维度分析，降为 v2 水平 |
| targeted_search 失败 | 跳过补搜，标注"信息缺口未填补" | 报告注明缺口 |
| echo_chamber 失败 | 跳过偏见检测 | 不影响核心内容 |
| fact_opinion 失败 | 不做分离，直接传递混合内容 | 报告结构降级 |
| synthesis 失败 | 直接传递辩论原始输出 | 报告质量降低 |
| Report Agent 失败 | 返回 synthesis JSON | 降级为数据展示 |

### 11.2 超时预算

| 阶段 | 超时 | 说明 |
|------|------|------|
| Phase 0（两 Agent 并行） | 300s | 取决于最慢的 Agent |
| Phase 1（桥接+分歧） | 30s | 含 1 次 LLM 调用 |
| Phase 2（辩论，含补搜） | 120s | 6 次 LLM 调用 + 可能的补搜 |
| Phase 3（偏见+事实分离） | 60s | 2 次 LLM 调用 |
| Phase 4（综合+报告） | 120s | 2 次 LLM 调用 + Report 生成 |
| **总计** | **~630s (~10.5min)** | |

### 11.3 LLM 调用优化

Phase 2 的 6 次 LLM 调用是延迟瓶颈。优化策略：
- Phase 2.1 的 4 个维度独立分析可以 **asyncio.gather 并行**（4 次 → 1 次延迟）
- 总实际延迟 = 1（并行独立分析）+ 1（交叉质证）+ 1（综合裁定）= 3 次 LLM 调用延迟 ≈ 15-30s

---

## 十二、实现路径与优先级

### Phase 1：基础串联 + 核心创新（最高优先级）

1. 搭建 `AgentCoordinator/` 目录结构
2. 实现 `query_agent_node` + `media_agent_node`（含超时 + 降级）
3. 实现 `data_bridge_node`（简化版，格式转换）
4. 实现 `deliberation_engine`（核心创新 1，从方案 C 开始：Phase 1 多调用 + Phase 2-3 单调用）
5. 实现 `synthesis_node`（MoA Aggregator）
6. 实现 `report_agent_node` + `report_bridge.py`
7. 集成测试：完整流程跑通

### Phase 2：深度创新

1. 实现 `divergence_matrix_node`（创新 5：分歧矩阵）
2. 实现 `echo_chamber_detector` + `fact_opinion_separator`（创新 4）
3. 实现 `gap_detector` + `targeted_search_node`（创新 2：CRAG 反馈补搜）
4. 实现 `platform_interpreter`（创新 3：平台画像解读）
5. 实现 `perspective_generator`（动态维度，创新 1 完善）

### Phase 3：报告呈现

1. 升级 Report Agent 模板支持新结构（创新 6）
2. 分歧矩阵热力图可视化
3. 辩论过程可折叠展示
4. 置信度标注系统
5. Flask WebSocket 进度推送

### Phase 4：生产化

1. LangGraph checkpointing（断点续传）
2. Report Fusion 时序一致性（多轮分析累积）
3. 性能调优（LLM 调用并行化、缓存）

---

## 十三、学术支撑汇总

### 核心论文

| 论文 | 支撑的创新点 | 关键贡献 |
|------|------------|---------|
| AI Council（arXiv:2604.26561，2025） | 创新1 辩论引擎 | 架构异构性保留分歧 > prompt engineering，120场实验验证 |
| MAD（Du et al.，EMNLP 2024） | 创新1 辩论引擎 | 多 Agent 迭代辩论改善推理质量 |
| RADAR（arXiv:2604.19005，2025） | 创新1 辩论引擎 | Politician/Scientist/Judge 三角质证结构 |
| NeurIPS 2025 "Debate or Vote" | 创新1 辩论引擎 | 纯 debate 不够，需要 biased belief update |
| CRAG（arXiv:2401.15884） | 创新2 动态补搜 | 检索质量评估 → 三路决策（correct/ambiguous/incorrect） |
| FAIR-RAG（arXiv:2510.22344） | 创新2 动态补搜 | 迭代检索精炼循环 + 结构化证据评估 |
| COLING 2025 Echo Chamber | 创新4 回声室破解 | LLM 驱动的社交网络极化检测 |
| MoA（ICLR 2025，arXiv:2406.04692） | 创新5 综合聚合 | Aggregator 真综合 > 选择 > 拼接 |
| Self-MoA（NeurIPS 2024） | 创新5 综合聚合 | Aggregator 质量 > Proposer 多样性 |
| MPSR（MDPI 2025） | 创新1+6 辩论+报告 | 多 stakeholder 视角 + Report Fusion 时序机制 |
| CSSD（本系统原创） | 创新5 分歧矩阵 | 跨源情感差异检测，从内部扩展到 Agent 间 |
| Anthropic 生产系统（2025） | 创新6 报告架构 | Artifact System 证据溯源 |
| CortexDebate（ACL 2025） | 创新1 辩论引擎 | 稀疏并行辩论提高效率 |
| Google DeepMind "Multi-Agent Trap" | 整体架构 | 无结构多 Agent 错误放大 17.2×，必须结构化拓扑 |
| ABSTRAL（arXiv:2603.22791） | 整体架构 | Fan-out + Aggregator 拓扑形式化验证 |

### 跨平台分析参考

| 来源 | 内容 |
|------|------|
| Nanjing Marketing Group 2026 | 知乎用户画像：71.5% 本科+，理性讨论平台 |
| Blue Lion Insight 2025 | B站用户画像：78.67% 90后/00后，均龄 22.8 |
| LinkinTech 2025 | 中国十大社媒平台对比分析 |
| CNNIC 2024 | 中国互联网用户规模与平台渗透率 |

---

*文档版本：v3.2 | 2026-05-02 | li_yewen*  
*v3.2 升级：创新6报告架构重写为 IMRaD 学术论文模式，新增 academic_report_generator.py 和 coordinator_output_schema.py，report_agent_node.py 重构*

---

## 十四、Phase 1 实现完成情况（2026-05-02）

### 14.1 完成状态

**Phase 1 全部节点已实现并端到端验证通过。**

| 文件 | 状态 | 说明 |
|------|------|------|
| `AgentCoordinator/graph/state.py` | ✅ | CoordinatorState TypedDict，含全4个阶段字段 |
| `AgentCoordinator/utils/timeout_guard.py` | ✅ | asyncio 超时封装 |
| `AgentCoordinator/utils/platform_profiles.py` | ✅ | 6平台画像（微博/知乎/B站/抖音/小红书/贴吧） |
| `AgentCoordinator/utils/perspective_templates.py` | ✅ | 6种分析类型 × 4维度模板（event/brand/policy/technology/general/person） |
| `AgentCoordinator/utils/report_bridge.py` | ✅ | synthesis_context → ReportAgent 格式转换 |
| `AgentCoordinator/prompts/deliberation_prompts.py` | ✅ | Phase 2.1/2.2/2.3 三阶段 Prompt |
| `AgentCoordinator/prompts/fact_separation_prompt.py` | ✅ | 事实-舆论分离 Prompt |
| `AgentCoordinator/prompts/synthesis_prompt.py` | ✅ | MoA 综合 Prompt |
| `AgentCoordinator/graph/nodes/query_agent_node.py` | ✅ | QueryEngine 调用 + 结果缓存（第一次运行后保存 JSON，后续复用） |
| `AgentCoordinator/graph/nodes/media_agent_node.py` | ✅ | MediaEngine 调用 + 测试数据注入（Media Agent 未启动时自动注入） |
| `AgentCoordinator/graph/nodes/data_bridge_node.py` | ✅ | 异构数据桥接，QueryAgentOutput + MediaAgent Markdown → BridgedProposition |
| `AgentCoordinator/graph/nodes/divergence_matrix_node.py` | ✅ | 跨源分歧矩阵（Innovation 5）：所有 Agent×平台 对 CSSD 计算 |
| `AgentCoordinator/graph/nodes/perspective_generator.py` | ✅ | 动态维度选择（根据 analysis_type） |
| `AgentCoordinator/graph/nodes/deliberation_engine.py` | ✅ | 3阶段辩论引擎（Innovation 1）：Phase 2.1 并行+Phase 2.2 交叉质证+Phase 2.3 综合裁定 |
| `AgentCoordinator/graph/nodes/gap_detector.py` | ✅ | CRAG 条件路由（Innovation 2） |
| `AgentCoordinator/graph/nodes/targeted_search_node.py` | ✅ | MindSpiderDB + Tavily 定向补搜 |
| `AgentCoordinator/graph/nodes/echo_chamber_detector.py` | ✅ | Shannon 熵计算 + 沉默大多数检测（Innovation 4） |
| `AgentCoordinator/graph/nodes/fact_opinion_separator.py` | ✅ | 事实-舆论-分析框架三层分离（Innovation 4） |
| `AgentCoordinator/graph/nodes/platform_interpreter.py` | ✅ | 平台画像解读（Innovation 3） |
| `AgentCoordinator/graph/nodes/synthesis_node.py` | ✅ | MoA 风格综合聚合（Innovation 5） |
| `AgentCoordinator/graph/nodes/report_agent_node.py` | ✅ | 学术风格报告生成（调用 academic_report_generator） |
| `AgentCoordinator/academic_report_generator.py` | ✅ | IMRaD 学术论文模式 Markdown 报告生成器（纯函数，确定性转换） |
| `AgentCoordinator/coordinator_output_schema.py` | ✅ | coordinator_output.json 完整 Schema + build_coordinator_output() |
| `AgentCoordinator/graph/builder.py` | ✅ | LangGraph 图构建，含 MemorySaver checkpointing + 条件回边 |
| `AgentCoordinator/coordinator.py` | ✅ | 统一入口（checkpointing + thread_id + coordinator_output.json 自动导出） |
| `AgentCoordinator/test_phase1.py` | ✅ | 端到端集成测试 |

### 14.2 验证结果（2026-05-02 实测）

**测试查询**：`"DeepSeek发布新模型 各方舆论"`

| 模块 | 指标 | 值 |
|------|------|-----|
| 总耗时（第2次运行，QueryAgent 命中缓存） | 102.9s | QueryAgent 0s + MediaAgent 0s + Deliberation 76s + Synthesis 11s |
| QueryAgent | 来源数 | 45 |
| QueryAgent | Coverage | 1.00 |
| 跨源分歧矩阵 | 来源数 | 6（query_agent, media_agent, weibo, bilibili, zhihu, social_media_overall） |
| 跨源分歧矩阵 | 对数 | 15 |
| 跨源分歧矩阵 | 最大分歧 | weibo vs zhihu: CSSD=1.000 |
| 跨源分歧矩阵 | 热点数 | 11（CSSD > 0.3） |
| 辩论引擎 | 维度数 | 4（Facts & Data / Public Emotion / Stakeholder Analysis / Historical Reflection） |
| 辩论引擎 | 共识点 | 7 条 |
| 辩论引擎 | 分歧点 | 6 条 |
| 辩论引擎 | 置信度 | 0.75 |
| 平台解读 | 覆盖平台 | bilibili, weibo, zhihu |
| 事实-舆论分离 | 验证事实 | 4 条 |
| 事实-舆论分离 | 观点/情感 | 3 条 |
| 事实-舆论分离 | 分析框架 | 3 条 |
| MoA 综合 | 置信度 | 0.72 |
| 报告 | 长度 | 8261 字符 |
| 报告 | 模式 | markdown_fallback（ReportEngine API Key 未配置） |

### 14.3 发现的问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `divergence_matrix` 节点名与 State 字段名冲突 | LangGraph 不允许节点名与 State key 同名 | 节点重命名为 `divergence_compute` |
| `gap_detector.py` 中 `from .state import CoordinatorState` 路径错误 | nodes/ 包内不能用 `.state` 导入 graph/state | 改为 `from ..state import CoordinatorState` |
| `platform_interpreter.py` 无法获取帖子数 | `per_platform` 结构使用 `count` 而非 `post_count` | 改为 `stats.get("post_count", 0) or stats.get("count", 0)` |
| ReportEngine 调用失败 | REPORT_ENGINE_API_KEY 未配置 | 自动降级为 Markdown 报告（预期行为） |

### 14.4 关键技术亮点（实测验证）

1. **并行执行**：QueryAgent + MediaAgent 并行启动（LangGraph superstep fan-out）；Phase 2.1 的 4 个维度分析也通过 asyncio.gather 并行，理论上 4 次调用时延 = 1 次调用时延
2. **结果缓存**：QueryAgent 结果在第一次成功运行后保存到 `AgentCoordinator/cache/query_agent_{hash}.json`，后续运行直接加载（0s）
3. **跨源分歧矩阵**：6 个来源 × 15 对 CSSD，成功发现 weibo vs zhihu CSSD=1.000（情绪化大众 vs 理性知识群体的立场截然相反）
4. **辩论引擎**：真正的 3 阶段结构化辩论，生成了有实质内容的跨维度共识和持续分歧

### 14.5 已升级项（Phase 4 中更新）

1. **report_agent_node.py 重构**：从"调用 ReportEngine 或 Markdown 降级"改为调用 `academic_report_generator.py` 的 `generate_academic_report()`，每次运行自动生成学术风格 Markdown 报告（IMRaD 模式，含摘要、方法论、研究发现、辩论分析、偏见评估、结论与附录）
2. **coordinator_output_schema.py 新增**：定义 coordinator_output.json 的完整 15 字段 Schema，每次运行自动导出 `coordinator_output_latest.json`
3. **builder.py 升级**：LangGraph MemorySaver checkpointing，支持断点续传
4. **coordinator.py 升级**：thread_id 隔离、checkpointing config、coordinator_output 自动导出

---

*文档版本：v3.2 | 2026-05-02 | li_yewen*
*Phase 1+2+4 实现完成，含学术风格报告生成器 + Checkpointing + coordinator_output.json Schema*

---

## 十五、Phase 2 & Phase 4 实现完成情况（2026-05-02）

### 15.1 Phase 2 说明

Phase 2 在设计文档中列为"深度创新节点"，实际上已在 Phase 1 实现时一并完成（设计与实现同步推进），具体状态：

| 创新节点 | 文件 | 状态 |
|---------|------|------|
| 跨源分歧矩阵（Innovation 5） | `graph/nodes/divergence_matrix_node.py` | ✅ Phase 1 完成 |
| 回声室检测+沉默大多数（Innovation 4 Layer1-2） | `graph/nodes/echo_chamber_detector.py` | ✅ Phase 1 完成 |
| 事实-舆论分离（Innovation 4 Layer3） | `graph/nodes/fact_opinion_separator.py` | ✅ Phase 1 完成 |
| CRAG 缺口检测路由（Innovation 2） | `graph/nodes/gap_detector.py` | ✅ Phase 1 完成 |
| CRAG 定向补搜（Innovation 2） | `graph/nodes/targeted_search_node.py` | ✅ Phase 1 完成 |
| 平台画像解读（Innovation 3） | `graph/nodes/platform_interpreter.py` | ✅ Phase 1 完成 |
| 动态维度生成（Innovation 1） | `graph/nodes/perspective_generator.py` | ✅ Phase 1 完成 |

**CRAG 补搜验证**（独立测试）：
- 构造含 `data_gaps` 的 state，`targeted_search_node` 成功触发 Tavily 搜索
- 1个信息缺口 → 5条补搜结果，耗时 3.3s

### 15.2 Phase 4 完成内容

**4.1 LangGraph Checkpointing（断点续传）**

修改 `AgentCoordinator/graph/builder.py`：
- 引入 `MemorySaver`，`build_coordinator_graph(use_checkpointing=True)` 
- 每次 `graph.compile(checkpointer=MemorySaver())` 自动开启检查点
- 支持 `use_checkpointing=False` 关闭（用于测试）

修改 `AgentCoordinator/coordinator.py`：
- `run(query, thread_id=None)` 每次生成唯一 `thread_id`（UUID4）
- 调用时传入 `config={"configurable": {"thread_id": thread_id}}`
- 节点级别失败后可从最后成功检查点恢复

**4.2 结构化输出 JSON（交给 ReportAgent 的中间产物）**

新增 `AgentCoordinator/coordinator_output_schema.py`（469行）：
- `COORDINATOR_OUTPUT_SCHEMA`：完整字段文档，每个 key 有类型、描述、来源说明
- `build_coordinator_output(result, query, duration)`：标准化构建输出 dict
- 每次 `run()` 结束后自动保存两份：
  - `AgentCoordinator/cache/coordinator_output_{timestamp}.json`（归档）
  - `AgentCoordinator/cache/coordinator_output_latest.json`（最新，供 ReportAgent 消费）

**实测输出 JSON（schema v1.0）验证结果**：

| 字段 | 值 |
|------|-----|
| `schema_version` | "1.0" |
| `divergence_matrix.pairs` 数量 | 15 对 |
| `divergence_matrix.hotspots` 数量 | 11 条 |
| `divergence_matrix.max_divergence` | weibo\|zhihu = 1.000 |
| `deliberation.perspectives_used` | 4 个维度 |
| `deliberation.final_consensus` | 9 条 |
| `deliberation.final_dissents` | 8 条 |
| `deliberation.confidence` | 0.70 |
| `fact_opinion_separation.verified_facts` | 6 条 |
| `fact_opinion_separation.opinions_sentiments` | 4 条 |
| `fact_opinion_separation.analytical_frameworks` | 3 条 |
| `synthesis.overall_confidence` | 0.70 |
| `synthesis.top_insights` | 3 条 |
| `synthesis.key_tensions` | 3 条 |
| `source_data.query_agent.total_sources` | 15（top，全量45） |
| `source_data.query_agent.coverage_score` | 1.00 |
| `source_data.social_sentiment.mode` | available |

---

## 十六、完整真实测试说明（首次运行 vs 命中缓存）

### 16.1 测试数据说明

| 数据来源 | 内容 | 位置 |
|---------|------|------|
| **QueryAgent 缓存** | 45条真实搜索结果，含 MindSpiderDB 社媒数据（weibo/zhihu/bilibili 23条帖子） | `AgentCoordinator/cache/query_agent_d109e3eef104.json` |
| **MediaAgent 测试数据** | 注入的模拟中文媒体报道（2270字符 Markdown） | `AgentCoordinator/graph/nodes/media_agent_node.py` 中 `_generate_test_media_data()` |
| **MindSpiderDB 真实数据** | MySQL `capstone` 库，weibo_note/zhihu_content/bilibili_video 等表 | 本地 MySQL，由 `QueryEngine/tools/mindspider_search.py` 读取 |

**Media Agent 说明**：Media Agent 当前配置 `_USE_TEST_DATA = True`（`media_agent_node.py` 第17行），注入测试数据。若需要真实运行，将其改为 `False` 并确保 MediaEngine API Key 已配置（需要 Gemini/Bocha Key）。

### 16.2 首次完整真实测试（从头搜索，不用缓存）

```bash
# 1. 进入项目目录（确保在服务器上执行）
cd /home/ubuntu/capstone/CapstoneProject

# 2. 删除缓存（强制重新搜索）
rm -f AgentCoordinator/cache/query_agent_*.json

# 3. 运行完整测试
.venv/bin/python AgentCoordinator/test_phase1.py

# 预期耗时：约 2-3 分钟（含 QueryAgent 真实搜索 ~35s + Deliberation ~75s）
```

**首次运行流程**：
1. QueryAgent 调用 Tavily API（国际新闻）+ Anspire（中文媒体）搜索
2. QueryAgent 查询 MindSpiderDB（MySQL `capstone` 库，读取 weibo_note/zhihu_content/bilibili_video）
3. MediaAgent 注入测试数据（不调用外部 API）
4. Coordinator 运行 Deliberation（6次LLM调用，其中4次并行）→ EchoChamber → FactOpinion → Synthesis
5. 结果自动保存：
   - `AgentCoordinator/cache/query_agent_{hash}.json`（QueryAgent缓存，下次0s加载）
   - `AgentCoordinator/cache/coordinator_output_latest.json`（交给ReportAgent的结构化JSON）
   - `AgentCoordinator/cache/test_report_{timestamp}.md`（当前降级Markdown报告）

### 16.3 后续快速测试（命中缓存，仅测试 Coordinator 逻辑）

```bash
cd /home/ubuntu/capstone/CapstoneProject

# QueryAgent 结果已缓存，只重跑 Coordinator 部分（~100s）
.venv/bin/python AgentCoordinator/test_phase1.py
```

### 16.4 单模块验证

```bash
# 验证 CRAG 补搜（targeted_search_node）
.venv/bin/python -c "
import asyncio, sys
sys.path.insert(0, '.')
from AgentCoordinator.graph.nodes.targeted_search_node import targeted_search_node
state = {
    'query': 'DeepSeek V4 benchmark scores',
    'search_rounds': 0,
    'deliberation_rounds': [{
        'phase': 'independent',
        'perspectives': [{'perspective': 'Technical Facts',
                          'data_gaps': ['What MMLU scores did DeepSeek V4 achieve?']}],
        'consensus_points': [], 'dissent_points': [],
    }],
    'coordinator_trace': [],
}
r = asyncio.run(targeted_search_node(state))
print('Results:', r['supplementary_results'][:1])
" 2>/dev/null

# 验证 MindSpiderDB 连接
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from QueryEngine.tools.mindspider_search import MindSpiderDB
db = MindSpiderDB()
r = db.search_topic_globally('AI', limit_per_table=2)
print('MindSpiderDB OK, total:', r.total)
" 2>/dev/null
```

---

## 十七、交给 ReportAgent 的结构化 JSON 规格说明

**文件位置**：`AgentCoordinator/cache/coordinator_output_latest.json`

**完整 Schema 定义**：`AgentCoordinator/coordinator_output_schema.py`

### 17.1 ReportAgent 应消费的核心字段

```
coordinator_output_latest.json
├── schema_version              "1.0"
├── query                       原始查询
├── analysis_type               event/brand/policy/technology/general
├── generated_at                ISO 时间戳
├── pipeline_duration_seconds   总耗时（秒）
│
├── divergence_matrix           ★ Innovation 5：跨源分歧矩阵
│   ├── pairs                   {source_a|source_b: CSSD_value, ...}  ← 热力图数据
│   ├── hotspots                [str, ...]                              ← 文字描述
│   ├── max_divergence          {pair: str, value: float}
│   └── min_divergence          {pair: str, value: float}
│
├── deliberation                ★ Innovation 1：多维度辩论引擎
│   ├── perspectives_used       [4个维度名称]
��   ├── phases                  [{phase, consensus_points, dissent_points}, ...]
│   ├── final_consensus         [str, ...]                              ← 跨维度共识
│   ├── final_dissents          [str, ...]                              ← 持续分歧
│   └── confidence              float 0-1
│
├── gap_filling                 ★ Innovation 2：CRAG 补搜
│   ├── rounds_performed        int（0=无需补搜）
│   ├── gaps_detected           [{description, source}, ...]
│   └── results_found           int
│
├── platform_interpretations    ★ Innovation 3：平台画像解读
│   ├── weibo                   str（含人群特征+立场解读）
│   ├── zhihu                   str
│   ├── bilibili                str
│   └── ...（其他有数据的平台）
│
├── bias_analysis               ★ Innovation 4：回声室+沉默大多数
│   ├── echo_warnings           [str, ...]
│   └── silent_majority_hypothesis   str | null
│
├── fact_opinion_separation     ★ Innovation 4：事实-舆论分离
│   ├── verified_facts          [{fact, sources, verification_status, confidence}, ...]
│   ├── opinions_sentiments     [{perspective, holders, sentiment_intensity, potential_biases}, ...]
│   └── analytical_frameworks   [{framework, analysis, certainty}, ...]
│
├── synthesis                   ★ MoA 综合
│   ├── summary                 str（执行摘要，3-5句）
│   ├── top_insights            [{insight, basis, confidence}, ...]
│   ├── key_tensions            [{tension, between, significance}, ...]
│   ├── overall_confidence      float
│   └── recommended_investigation   [str, ...]
│
├── source_data                 原始 Agent 数据摘要
│   ├── query_agent             {total_sources, stance_distribution, coverage_score, top_sources, social_sentiment}
│   └── media_agent             {available, mode, summary_length}
│
├── coordinator_trace           [str, ...]（完整执行日志）
└── agent_errors                [str, ...]
```

### 17.2 ReportAgent 使用建议

| 报告章节 | 使用字段 |
|---------|---------|
| 执行摘要 | `synthesis.summary` + `synthesis.overall_confidence` |
| 事实层 | `fact_opinion_separation.verified_facts` |
| 舆论地形图 | `source_data.query_agent.stance_distribution` + `source_data.query_agent.social_sentiment` |
| 分歧矩阵热力图 | `divergence_matrix.pairs`（用来渲染热力图） |
| 多维度辩论 | `deliberation.phases` + `deliberation.final_consensus` + `deliberation.final_dissents` |
| 平台解读 | `platform_interpretations` |
| 偏见声明 | `bias_analysis.echo_warnings` + `bias_analysis.silent_majority_hypothesis` |
| 分析框架 | `fact_opinion_separation.analytical_frameworks` |
| 关键来源 | `source_data.query_agent.top_sources` |
| 置信度标注 | 各层级 `confidence` 字段 |

---

*文档版本：v3.2 | 2026-05-02 | li_yewen*  
*Phase 2（已合并在Phase1实现）+ Phase 4（Checkpointing + Coordinator Output JSON）全部完成*
