# AgentCoordinator 开发报告

> 作者：li_yewen | 日期：2026-05-02  
> 本文档面向项目组成员，说明 AgentCoordinator 做了什么、当前状态、如何测试、如何与 Report Agent 对接。

---

## 一、做了什么

AgentCoordinator 是整个系统的**核心协调层**，位于 QueryAgent 和 MediaAgent **之后**、ReportAgent **之前**。它不是简单地拼接两个 Agent 的结果，而是对多源数据进行**深度分析和再加工**。

### 输入

| 来源 | 格式 | 说明 |
|------|------|------|
| QueryAgent | `QueryAgentOutput` (TypedDict) | 结构化数据：45条来源、立场分布、社媒情感（CSSD、各平台分布）、意见聚类 |
| MediaAgent | Markdown 文本 | 中文媒体报道分析（当前 MediaAgent 未启动，使用注入测试数据） |

### 做了什么（6 大创新）

| # | 创新点 | 说明 | 实现文件 |
|---|--------|------|---------|
| 1 | **多维度辩论引擎** | 4个维度独立分析 → 交叉质证 → 综合裁定，模拟学术审议 | `deliberation_engine.py` |
| 2 | **CRAG 动态补搜** | 辩论发现信息缺口 → 自动搜索补充 → 回到辩论 | `gap_detector.py` + `targeted_search_node.py` |
| 3 | **平台画像解读** | 微博/知乎/B站等平台的用户人群特征 × 立场差异 = 社会结构洞察 | `platform_interpreter.py` |
| 4 | **回声室破解 + 事实分离** | Shannon 熵检测回声室、沉默大多数假设、事实-观点-分析框架三层分离 | `echo_chamber_detector.py` + `fact_opinion_separator.py` |
| 5 | **跨源分歧矩阵** | 所有数据源两两计算 CSSD（cosine 距离），发现结构性分歧 | `divergence_matrix_node.py` |
| 6 | **学术风格报告** | IMRaD 模式 Markdown：摘要 → 方法论 → 研究发现 → 辩论 → 结论 → 附录 | `academic_report_generator.py` |

### 输出

| 输出物 | 路径 | 用途 |
|--------|------|------|
| **coordinator_output_latest.json** | `AgentCoordinator/cache/coordinator_output_latest.json` | ★ 交给 ReportAgent 的结构化 JSON（15个字段，~40KB） |
| **test_report_{ts}.md** | `AgentCoordinator/cache/test_report_*.md` | 当前的学术风格 Markdown 报告（中间展示用） |
| **coordinator_output_{ts}.json** | `AgentCoordinator/cache/coordinator_output_*.json` | 归档 |

---

## 二、当前状态

| 项目 | 状态 |
|------|------|
| Phase 1（基础串联 + 核心创新） | ✅ 完成 |
| Phase 2（深度创新节点） | ✅ 完成（合并在 Phase 1） |
| Phase 4（Checkpointing + JSON Schema + 报告生成器） | ✅ 完成 |
| Phase 3（报告呈现：热力图可视化、辩论交互、WebSocket 进度）| ⏳ 留给 ReportAgent 实现 |
| QueryAgent 集成 | ✅ 真实调用，结果已缓存 |
| MediaAgent 集成 | ⚠️ 使用测试数据（MediaAgent 未启动） |
| MindSpider 社媒数据 | ✅ 真实读取 MySQL capstone 库（weibo/zhihu/bilibili） |

---

## 三、如何测试

### 3.1 快速测试（命中 QueryAgent 缓存，~100s）

```bash
cd /home/ubuntu/capstone/CapstoneProject
.venv/bin/python AgentCoordinator/test_phase1.py
```

QueryAgent 结果已缓存（`cache/query_agent_*.json`），不会重复调用 API。只跑 Coordinator 部分（辩论 ~75s + 综合 ~15s）。

### 3.2 完整真实测试（从头搜索，~3min）

```bash
cd /home/ubuntu/capstone/CapstoneProject
rm -f AgentCoordinator/cache/query_agent_*.json   # 删缓存
.venv/bin/python AgentCoordinator/test_phase1.py
```

会真实调用 Tavily API 搜索 + MindSpider 查 MySQL。

### 3.3 结果在哪里看

运行后终端会打印测试摘要。详细结果：

| 文件 | 内容 |
|------|------|
| `cache/test_report_*.md` | ★ **Markdown 报告**（直接用编辑器/GitHub 预览打开看） |
| `cache/coordinator_output_latest.json` | 结构化 JSON（给 ReportAgent 用） |
| `cache/test_results_*.json` | 完整运行结果（含 deliberation_rounds 等大量细节） |

### 3.4 测试数据说明

- **QueryAgent 数据**：真实搜索结果（45条来源）+ 真实 MindSpider 社媒数据（23条帖子、14条评论，来自 weibo/zhihu/bilibili）
- **MediaAgent 数据**：注入测试数据（`media_agent_node.py` 中 `_generate_test_media_data()` 函数生成的 2270 字符模拟 Markdown）。如需切换为真实 MediaAgent：将 `AgentCoordinator/graph/nodes/media_agent_node.py` 第 17 行 `_USE_TEST_DATA = True` 改为 `False`，并确保 MediaEngine API Key 已配置

---

## 四、交付给 Report Agent 的是什么

### 4.1 文件

`AgentCoordinator/cache/coordinator_output_latest.json`

每次 `coordinator.run()` 后自动覆盖写入。ReportAgent 直接读这个文件即可。

### 4.2 JSON 结构（15 个顶层字段）

```
coordinator_output_latest.json
├── schema_version          "1.0"
├── query                   "DeepSeek发布新模型 各方舆论"
├── analysis_type           "general"
├── generated_at            "2026-05-02T23:02:49"
├── pipeline_duration_seconds   125.3
│
├── divergence_matrix           ← Innovation 5
│   ├── pairs                   {"weibo|zhihu": 1.0, ...}  共 15 对
│   ├── hotspots                ["weibo vs zhihu: CSSD=1.000", ...]
│   ├── max_divergence          {"pair": "weibo|zhihu", "value": 1.0}
│   └── min_divergence          {"pair": "bilibili|social_media_overall", "value": 0.003}
│
├── deliberation                ← Innovation 1
│   ├── perspectives_used       ["Facts & Data", "Public Emotion", ...]
│   ├── phases                  [3 个阶段的 consensus/dissent]
│   ├── final_consensus         [8 条]
│   ├── final_dissents          [8 条]
│   └── confidence              0.80
│
├── gap_filling                 ← Innovation 2
│   ├── rounds_performed        0
│   ├── gaps_detected           []
│   └── results_found           0
│
├── platform_interpretations    ← Innovation 3
│   ├── bilibili                "**Bilibili (B站)** (18 posts)..."
│   ├── weibo                   "**Weibo (微博)** (3 posts)..."
│   └── zhihu                   "**Zhihu (知乎)** (2 posts)..."
│
├── bias_analysis               ← Innovation 4
│   ├── echo_warnings           []
│   └── silent_majority_hypothesis  null
│
├── fact_opinion_separation     ← Innovation 4
│   ├── verified_facts          [6 条，含 verification_status + confidence]
│   ├── opinions_sentiments     [3 条，含 holders + potential_biases]
│   └── analytical_frameworks   [3 条，含 framework + certainty]
│
├── synthesis
│   ├── summary                 "公众舆论呈现强烈的官方-民间叙事共振..."
│   ├── top_insights            [3 条，各含 insight + basis + confidence]
│   ├── key_tensions            [3 条，各含 tension + between + significance]
│   ├── overall_confidence      0.65
│   └── recommended_investigation  ["...", "...", "..."]
│
├── source_data
│   ├── query_agent             {total_sources, stance_distribution, top_sources, social_sentiment}
│   └── media_agent             {available, mode, summary_length}
│
├── coordinator_trace           [11 条执行日志]
└── agent_errors                []
```

完整 Schema 定义：`AgentCoordinator/coordinator_output_schema.py`

### 4.3 Report Agent 可以用这些数据做什么

| 报告章节 | 使用字段 | 怎么做 |
|---------|---------|--------|
| 分歧矩阵热力图 | `divergence_matrix.pairs` | Plotly/Seaborn 渲染 6×6 热力图 |
| 辩论过程可折叠展示 | `deliberation.phases` | HTML `<details>` 或 JS 折叠组件 |
| 置信度标注 | 各层 `confidence` 字段 | 每章节旁标 ★★★☆☆ |
| 事实层渲染 | `fact_opinion_separation.verified_facts` | ✅/⚠️ 图标 + 来源链接 |
| 平台画像卡片 | `platform_interpretations` | 每平台一个卡片，含人群画像 + 立场分布 |
| 社媒帖子引用 | `source_data.query_agent.social_sentiment.top_social_voices` | 原文引用 + 🔗跳转链接 |
| 高赞热评 | `source_data.query_agent.social_sentiment.comment_sentiment.top_comments` | 按点赞数排序展示 |
| WebSocket 进度 | `coordinator_trace` | 实时推送各节点状态 |

---

## 五、如何调用 AgentCoordinator

### 5.1 Python 调用

```python
import asyncio
import sys
sys.path.insert(0, '.')

from AgentCoordinator.coordinator import AgentCoordinator

async def main():
    coordinator = AgentCoordinator()
    result = await coordinator.run("DeepSeek发布新模型 各方舆论")
    
    # Markdown 报告
    print(result["report_output"])
    
    # 结构化 JSON 已自动保存到
    # AgentCoordinator/cache/coordinator_output_latest.json
    print(f"JSON saved to: {result['coordinator_output_path']}")

asyncio.run(main())
```

### 5.2 同步调用（非 async 环境）

```python
coordinator = AgentCoordinator()
result = coordinator.run_sync("某个话题")
```

---

## 六、Report Agent 对接指南

### 6.1 当前状态（Markdown 中间报告）

AgentCoordinator 的 `report_agent_node` 当前生成 **学术风格 Markdown**，其中有多个 `[PLACEHOLDER]` 标记，标注了 ReportAgent 需要实现的可视化/交互功能：

```
[VISUALIZATION PLACEHOLDER: 跨源分歧矩阵热力图]
[PLACEHOLDER: 可折叠辩论过程交互式展示]
[PLACEHOLDER: 置信度标注系统]
[FEATURE PLACEHOLDER: Flask WebSocket 实时进度]
```

### 6.2 如何直接 Pipeline 对接 ReportAgent

ReportAgent 需要做的：

```python
# 在 ReportAgent 中：
import json

# 1. 读取 AgentCoordinator 的结构化输出
with open("AgentCoordinator/cache/coordinator_output_latest.json") as f:
    coordinator_output = json.load(f)

# 2. 从中提取需要的数据
divergence_pairs = coordinator_output["divergence_matrix"]["pairs"]        # 热力图数据
deliberation = coordinator_output["deliberation"]                          # 辩论过程
facts = coordinator_output["fact_opinion_separation"]["verified_facts"]     # 事实层
social_voices = coordinator_output["source_data"]["query_agent"]["social_sentiment"]["top_social_voices"]  # 社媒帖子

# 3. 渲染为 HTML
# - 用 Plotly 画热力图
# - 用 <details> 做可折叠辩论
# - 用置信度数据标注各章节
```

### 6.3 如果要在 LangGraph Pipeline 中直接串联

修改 `AgentCoordinator/graph/nodes/report_agent_node.py`，将 `generate_academic_report()` 替换为调用你的 ReportAgent：

```python
# 当前：
from ...academic_report_generator import generate_academic_report
report = generate_academic_report(generator_input)

# 替换为：
from ReportEngine.agent import ReportAgent
agent = ReportAgent()
report = agent.generate_report(
    query=state["query"],
    reports=[generator_input],   # 把整个结构化 dict 传进去
)
```

需要确保 `REPORT_ENGINE_API_KEY` 在 `.env` 中配置（ReportEngine 使用 Gemini）。

---

## 七、目录结构

```
AgentCoordinator/
├── coordinator.py                      # 统一入口
├── coordinator_output_schema.py        # JSON Schema 定义
├── academic_report_generator.py        # 学术风格报告生成器
├── test_phase1.py                      # 端到端测试脚本
├── graph/
│   ├── builder.py                      # LangGraph 图（13 节点 + CRAG 回边）
│   ├── state.py                        # CoordinatorState TypedDict
│   └── nodes/                          # 13 个节点实现
├── utils/                              # 平台画像、维度模板、超时封装
├── prompts/                            # 辩论/分离/综合 Prompt
└── cache/                              # 运行时产物
    ├── coordinator_output_latest.json  # ★ 交给 ReportAgent 的 JSON
    ├── test_report_*.md                # 示例 Markdown 报告
    └── query_agent_*.json              # QueryAgent 结果缓存
```

---

*文档版本：v1.0 | 2026-05-02 | li_yewen*
