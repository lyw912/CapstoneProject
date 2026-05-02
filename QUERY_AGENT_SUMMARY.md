# Query Agent — 开发总结（v3.1）

> 作者：li_yewen | 更新：2026-05-02
> 范围：QueryEngine v2（LangGraph 重构）+ Phase 3 MindSpider 社媒数据集成

---

## 一、项目概览

**BettaFish**（微舆）是一个多智能体舆情分析平台，上游开源项目：`https://github.com/666ghj/BettaFish`。

### 五层架构

```
Layer 1: 用户界面层
  Flask 主应用（app.py, port 5000）
  + 三个 Streamlit 子应用（SingleEngineApp/, port 8501-8503）

Layer 2: 多 Agent 分析层
  QueryEngine   — Tavily 网络搜索 + DeepSeek 推理
  MediaEngine   — Bocha/Anspire 中文多模态搜索 + Gemini 2.5 Pro
  InsightEngine — MySQL/PostgreSQL 社媒数据库查询 + Kimi K2（500K上下文）
  ForumEngine   — LLM 主持人协调（Qwen Plus），文件总线通信

Layer 3: 报告生成层
  ReportEngine  — 模板选择 → 布局规划 → 章节生成 → HTML/PDF/Markdown

Layer 4: 数据采集层
  MindSpider    — 后台爬虫，覆盖 6 个社媒平台
                  （小红书 / 抖音 / B站 / 微博 / 贴吧 / 知乎）
                  注：快手已移除（与抖音内容重叠）

Layer 5: 存储与外部服务
  MySQL（社媒数据，capstone 数据库）
  Tavily / Bocha / Anspire API（搜索）
  各 LLM API（DeepSeek / Gemini / Kimi / Qwen）
```

---

## 二、Query Agent v2 — LangGraph 重构

### 2.1 与原项目 QueryEngine 的对比

| 维度 | 原项目 QueryEngine | Query Agent v2 |
|:-----|:------------------|:---------------|
| 架构 | 固定线性流水线，6节点串行 | LangGraph 子图，8节点含条件边和循环 |
| 搜索策略 | 单源 Tavily | 多源并行（Tavily + Anspire + MindSpider） |
| 立场意识 | 无 | 5维立场矩阵（official/support/oppose/neutral/background） |
| 来源评估 | 无，等权 | TrustScore 4维评分（域名权威+时效+内容质量+搜索排名） |
| 去重 | 无 | URL精确去重 + MinHash LSH内容去重 |
| 终止条件 | 固定轮次 | SCS驱动的自适应终止 |
| 输出格式 | 非结构化 Markdown | 结构化 `QueryAgentOutput`（JSON） |

### 2.2 图拓扑（Phase 3）

```
START → query_planner → unified_search → dedup_filter → trust_scorer
  → stance_classify → social_enrichment → coverage_check → [router]
                                                             ├─ output_assemble → END
                                                             └─ gap_filler → unified_search
```

### 2.3 新增目录结构

```
QueryEngine/
├── graph/
│   ├── state.py                # LangGraph 状态定义（TypedDict）
│   ├── builder.py              # 图构建，含条件边 coverage_router
│   └── nodes/
│       ├── query_planner.py    # 立场矩阵子查询规划
│       ├── unified_search.py   # 多源并行搜索
│       ├── dedup_filter.py     # URL去重 + MinHash LSH
│       ├── trust_scorer.py     # TrustScore 4维评分
│       ├── stance_classify.py  # 立场分类
│       ├── social_enrichment.py  # Phase 3：MindSpider 集成节点
│       ├── coverage_check.py   # SCS 计算 + 路由决策
│       ├── gap_filler.py       # 缺口补搜
│       └── output_assemble.py  # 结构化输出组装
├── classifiers/
│   ├── trust_scorer.py         # 4维TrustScore实现
│   └── stance_classifier.py    # 规则+关键词+LLM混合分类
├── fusion/
│   ├── rrf.py                  # Reciprocal Rank Fusion（SIGIR 2009）
│   └── dedup.py                # MinHash LSH去重
├── tools/
│   ├── search_dispatcher.py    # 统一调度 Tavily/Anspire/MindSpider
│   └── mindspider_search.py    # Phase 3：MindSpiderDB 搜索客户端
└── evaluation/
    ├── metrics.py              # SCS/SDI/SBS/TSM 计算
    ├── test_queries.py         # 20条标准测试集
    └── run_evaluation.py       # CLI评估脚本
```

### 2.4 核心算法

#### 立场矩阵子查询规划

LLM 在规划阶段生成 5–8 个子查询，每个标注目标立场维度。`_ensure_stance_coverage()` 兜底保证即使 LLM 漏了某立场也会补上。在上游解决多样性，而非下游重排。

**文献依据**：Draws et al.（SIGIR 2021）— 搜索结果立场偏差测量；MMR/xQuAD 多样性。

#### SCS 驱动的自适应终止

```
SCS = (1/K) × Σ min(count(stance_k) / threshold_k, 1.0)
thresholds = {support:2, oppose:2, official:1, neutral:1}

SCS < 1.0 且轮次 < 3  →  GapFiller 生成补搜  →  回到搜索
SCS = 1.0 或轮次 = 3  →  强制输出
```

**文献依据**：Self-RAG（ICLR 2024），CRAG（arXiv 2401.15884），Adaptive-RAG（arXiv 2403.14403）。

#### TrustScore

```python
score = 0.30 * domain_authority    # 60+ 个域名权威性字典
      + 0.25 * timeliness          # 7天半衰期指数衰减
      + 0.25 * content_quality     # snippet 长度 + 是否有全文
      + 0.20 * rrf_score           # 搜索 API 相关性得分
```

#### 权威来源优先去重

MinHash LSH 去重时，若两条内容被判定为重复，保留来自权威域名（`.gov.cn`、`xinhua.net` 等）的那条，而非简单保留先出现的。

---

## 三、Phase 3 — MindSpider 社媒数据集成

### 3.1 设计目标

1. **双层信息获取**：Tavily/Anspire 提供网络/媒体叙事层，MindSpider 提供社媒情感层。
2. **跨源情感差异检测（CSSD）**：自动发现网络搜索结果与社媒讨论之间的立场分布差异。
3. **自适应降级**：MindSpider 无数据时 `social_sentiment` 返回 `null`，系统行为与改造前完全一致。
4. **来源可追溯**：所有社媒数据展示平台、URL、发布时间，确保不是编造。

### 3.2 学术参考

| 技术 | 来源 | 在系统中的角色 |
|------|------|--------------|
| Resource Selection | Callan et al., CORI, SIGIR 1995 | Probe-then-decide：先 COUNT 探测再决定是否全量查询 |
| Corrective RAG (CRAG) | Yan et al., 2024 | 社媒数据作为网络搜索结果的"校正源"，交叉验证 |
| Adaptive-RAG | Jeong et al., 2024 | 根据数据可用性动态选择检索策略 |
| RRF | Cormack et al., SIGIR 2009 | 跨源排名融合（已有） |
| Stance Detection | Mohammad et al., SemEval 2016 | 立场分类器扩展到跨源对比 |

### 3.3 social_enrichment 节点逻辑

```
1. 从查询中提取关键词（分离英文 token 和中文词）
2. 逐关键词探测 MindSpider（COUNT 查询，<50ms）
3. 判断模式：
   - total_posts < 3        → disabled（纯 API 模式）
   - freshness < 72h        → available（完整混合模式）
   - freshness >= 72h       → stale（降权使用）
4. 全量查询：获取社媒帖子
5. LLM 批量立场分类（扩展4）
6. 计算 CSSD 分数
7. LLM 生成跨源对比摘要
8. 选取 Top 10 代表性社媒声音
9. 评论情感聚合（扩展1）
10. 时序情感追踪（扩展2）
11. 数据过期/缺失时触发 BTE（扩展3，fire-and-forget）
```

### 3.4 CSSD 计算公式

```
CSSD = 1 - cosine_similarity(web_stance_vector, social_stance_vector)

stance_vector = [support_ratio, oppose_ratio, neutral_ratio, official_ratio, background_ratio]

CSSD = 0：分布完全一致
CSSD = 1：完全相反
CSSD > 0.5：显著差异（系统发出告警）
```

### 3.5 扩展1 — 评论情感聚合

帖子是表面，评论区才是深层民意。`MindSpiderDB.search_comments()` 跨 6 个评论表搜索，用 LLM 批量分类器对评论进行立场分类，按点赞数排序展示 Top 评论。

输出字段：`social_sentiment.comment_sentiment`

```json
{
  "total": 6,
  "distribution": {"support": 0.667, "neutral": 0.333},
  "top_comments": [
    {"platform": "weibo", "content": "...", "like_count": 567, "stance": "support"}
  ]
}
```

### 3.6 扩展2 — 时序情感追踪

`MindSpiderDB.search_with_time_buckets()` 按日期分桶查询（7天窗口）。对每个时间桶内的帖子进行立场分类，通过比较前半段和后半段的 support 比例变化（delta > 0.1 判定为 rising/falling）检测趋势。已分类的帖子通过 content→stance 查找表复用，避免重复 LLM 调用。

输出字段：`social_sentiment.sentiment_trend`

```json
{
  "buckets": [{"date": "2026-05-01", "post_count": 23, "distribution": {...}}],
  "trend_direction": "stable",
  "trend_summary": "..."
}
```

### 3.7 扩展3 — 主动触发 BroadTopicExtraction

当模式为 `disabled` 或 `stale` 且 `daily_topics` 表今天无数据时，通过 `subprocess.Popen(start_new_session=True)` 启动独立进程运行 BroadTopicExtraction（约30秒，无 Playwright）。当前查询不等待结果，继续执行。

### 3.8 扩展4 — LLM 批量立场分类器

`HybridStanceClassifier.classify_batch_llm()` 单次 LLM 调用批量分类所有帖子，返回 JSON 数组。失败时自动降级为逐条规则分类。LLM 分类器能识别规则分类器无法处理的反讽、隐含情感和网络用语。

**同一批 23 条社媒帖子的效果对比：**

| 分类器 | support | oppose | neutral | background |
|--------|---------|--------|---------|------------|
| 规则分类 | 17.4% | 13.0% | 65.2% | 4.3% |
| LLM 批量 | 26.1% | 17.4% | 47.8% | 8.7% |

### 3.9 质量保障措施

**水军/刷评检测**：内容多样性分数 = `unique_posts / total_posts`。< 0.7 且帖子数 ≥ 5 时触发告警："结果可能受协调发帖影响"。

**政治风险规避**：对比框架为"网络搜索结果 vs 社媒讨论"，不使用"官方 vs 民意"表述。所有 LLM Prompt 加约束：`Do NOT frame this as "official vs public" or imply any political narrative`。

**平台粒度分析**：按平台分别计算立场分布，揭示用户群差异（微博：大众/情绪化；知乎：高学历/理性；B站：年轻/多元）。

**性能优化**：补搜循环时检测 `social_sentiment` 已存在则跳过重复执行；时序分析复用已分类帖子。总耗时从 73s 降至 29.6s（-59%）。

### 3.10 输出格式

`QueryAgentOutput` 新增 `social_sentiment` 字段：

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
    "divergence_summary": "网络搜索结果以支持和背景立场为主，社媒讨论分布更均衡且有明显反对声音。",
    "freshness_hours": 0.5,
    "top_social_voices": [...],
    "comment_sentiment": {"total": 6, "distribution": {...}, "top_comments": [...]},
    "sentiment_trend": {"buckets": [...], "trend_direction": "stable", "trend_summary": "..."},
    "crawl_triggered": false
  }
}
```

MindSpider 无数据时，`social_sentiment` 为 `null`，系统行为与改造前完全一致。

### 3.11 验证结果（2026-05-01）

测试查询："DeepSeek发布新模型 各方舆论"

| 指标 | 值 |
|------|-----|
| 社媒模式 | available |
| 覆盖平台 | weibo, zhihu, bilibili |
| 社媒帖子数 | 23 |
| 社媒评论数 | 6 |
| CSSD 分数 | 0.181 |
| 内容多样性 | 82.6%（健康） |
| 趋势方向 | stable |
| 总耗时 | 29.6s |
| 保留来源数 | 53 |

---

## 四、MindSpider 部署与运维

### 4.1 架构

```
MindSpider/
├── BroadTopicExtraction/    # 阶段一：话题发现（轻量，无需登录）
└── DeepSentimentCrawling/   # 阶段二：深度爬取（Playwright，需要 Cookie）
    └── MediaCrawler/        # 实际爬虫核心（浏览器自动化）
```

**BroadTopicExtraction**：调用公开新闻聚合 API（12个平台），用 DeepSeek 提取关键词，写入 `daily_news` 和 `daily_topics` 表。无需登录，无 Playwright，约30秒，资源消耗极低。

**DeepSentimentCrawling**：用 Playwright 模拟浏览器，爬取帖子正文和评论。需要各平台登录 Cookie。写入 `weibo_note`、`zhihu_content` 等平台表。

### 4.2 数据库

**数据库：`capstone`**（与主项目共用同一 MySQL 数据库，非独立的 `mindspider` 库）

包含：
- 核心表：`daily_news`、`daily_topics`、`topic_news_relation`、`crawling_tasks`
- 平台内容表：`xhs_note`、`douyin_aweme`、`bilibili_video`、`weibo_note`、`tieba_note`、`zhihu_content`
- 评论表：`weibo_note_comment`、`zhihu_comment`、`bilibili_video_comment`、`douyin_aweme_comment`、`xhs_note_comment`、`tieba_comment`

### 4.3 定时爬取

文件：`MindSpider/scheduled_run.sh`
Cron：`35 2 * * *`（每天凌晨 02:35）

```
Step 1: BroadTopicExtraction（约30秒，无 Playwright）
Step 2: Tier 1 平台每天运行 — 微博 → 知乎 → B站（--max-notes 20）
Step 3: Tier 2 平台仅奇数日 — 小红书 → 抖音 → 贴吧（--max-notes 10）
```

内存保护：每个平台启动前检查可用 RAM（< 200MB 等30s，仍不足则跳过），跑完后强制释放浏览器进程，锁文件防止 cron 重叠触发。

**持久化日志**：
- 定时任务日志：`MindSpider/logs/scheduled_run_YYYYMMDD.log`（按日期分文件）
- 手动全量日志：`MindSpider/logs/full_run_YYYYMMDD_HHMMSS.log`
- 临时实时日志：`/tmp/mindspider.log`

### 4.4 Cookie 管理

文件：`cookie.txt`（项目根目录，不提交——含凭据）
格式：每行一个平台，如 `weibo=<cookie字符串>`

`MindSpider/DeepSentimentCrawling/platform_crawler.py` 在每次爬取前自动从 `cookie.txt` 读取对应平台 cookie，写入 `MediaCrawler/config/base_config.py`，无需手动修改配置文件。

2026-05-02 Cookie 状态（全量运行实测：6/6 PASS，含 xhs 更新后）：

| 平台 | 状态 | 预计有效期 | 备注 |
|------|------|-----------|------|
| 微博 | ✅ 有效 | 2026-07-27（约85天） | 需从 m.weibo.cn 获取 |
| 知乎 | ✅ 有效 | SESSIONID 数周 | z_c0 自动刷新，无需手动操作 |
| B站 | ✅ 有效 | 2026-10-28（约179天） | 走 bilibili-api-python，绕开 Playwright 风控 |
| 小红书 | ✅ 有效 | ~30天滚动 | 风控严，触发后手机端也会被踢出，需定期检查 |
| 抖音 | ✅ 有效 | 2026-06-30（约60天） | sid_guard 到期需更新 |
| 贴吧 | ✅ 有效 | 6个月以上 | BDUSS 长期有效 |

**微博注意**：MediaCrawler 使用 `m.weibo.cn`（移动端），必须从 `https://m.weibo.cn` 获取 cookie，从 `weibo.com`（桌面端）获取的无效。

**知乎 z_c0 自动刷新**：`DeepSentimentCrawling/zhihu_cookie_refresher.py` 在每次知乎爬取前用 Playwright 加载知乎首页，利用长效的 `SESSIONID` 自动刷新 `z_c0` 并写回 `cookie.txt`，无需手动操作。

### 4.5 去重机制

各平台 store 层均采用**"先查询，后更新或插入"**模式（以 `note_id`/`comment_id` 为唯一键）：
- 已存在 → 只更新 `last_modify_ts`，**不重复插入**
- 不存在 → INSERT，记录 `add_ts`（首次入库时间）

即使同一话题长期在热榜，旧帖子和旧评论不会重复写入，只有真正新增的内容才会入库。

### 4.6 监控脚本

文件：`MindSpider/monitor.py`
Cron：`7 8 * * *`（每天早上 08:07）
输出：`/tmp/mindspider_alerts.log`

| 检测项 | 阈值 | 告警标签 |
|--------|------|---------|
| Cookie 有效性（live HTTP） | 失败 | `[COOKIE]` |
| Cookie 即将过期 | ≤ 3天 | `[COOKIE]` |
| 爬取日志 cookie 失效关键词 | 401/403/登录状态已失效 | `[COOKIE_LOG]` |
| 定时任务启动但未完成 | 有 started 无 finished | `[CRAWL]` |
| 内存使用率 | > 85% | `[RESOURCE]` |
| CPU 使用率 | > 90%（2秒采样） | `[RESOURCE]` |
| 磁盘使用率 | > 80% | `[DISK]` |

退出码：0 = 全部正常，1 = 有告警触发。

### 4.7 服务器资源限制

服务器：2核 CPU，2GB 内存，40GB 硬盘。

Playwright 每个浏览器实例约占 200–400MB，必须串行爬取，不能并发。推荐：每次只跑1个平台，每天定时一次，`--max-notes 20` 控制单次运行在10分钟内。

---

## 五、Phase 3 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `QueryEngine/graph/state.py` | 修改 | 新增 `"mindspider_db"` 到 `SubQueryItem.target_source`；新增 `social_sentiment`、`mindspider_mode` 字段 |
| `QueryEngine/graph/nodes/gap_filler.py` | 修改 | `"insight_db"` → `"mindspider_db"` |
| `QueryEngine/graph/nodes/query_planner.py` | 修改 | Prompt 更新；fallback 路由使用 mindspider_db |
| `QueryEngine/tools/mindspider_search.py` | 新增 | `MindSpiderDB`，含 `probe()`、`search_comments()`、`search_with_time_buckets()`、`has_extraction_today()` |
| `QueryEngine/graph/nodes/social_enrichment.py` | 新增 | 社媒增强节点（含全部4个扩展 + 质量保障） |
| `QueryEngine/classifiers/stance_classifier.py` | 修改 | 新增 `classify_batch_llm()` 和 `_parse_batch_response()` |
| `QueryEngine/classifiers/trust_scorer.py` | 修改 | mindspider_db 源加 +0.05 可信度补偿 |
| `QueryEngine/graph/builder.py` | 修改 | 注册 social_enrichment 节点；重连 stance_classify → social_enrichment → coverage_check |
| `QueryEngine/graph/nodes/__init__.py` | 修改 | 导出 social_enrichment_node |
| `QueryEngine/tools/__init__.py` | 修改 | 导出 MindSpiderDB、MindSpiderResponse、MindSpiderResult、MindSpiderComment |
| `QueryEngine/agent.py` | 修改 | 初始状态包含 mindspider_mode 和 social_sentiment 字段 |
| `QueryEngine/graph/nodes/output_assemble.py` | 修改 | 输出包含 social_sentiment |
| `SingleEngineApp/query_agent_temp_app.py` | 修改 | 新增第5个 Tab "Social Sentiment"，含所有 Phase 3 可视化 |
| `tests/sample_data.sql` | 新增 | 23条社媒示例帖子（微博/知乎/B站，DeepSeek话题）用于测试 |
| `tests/sample_comments.sql` | 新增 | 20条示例评论（微博/知乎）用于评论情感测试 |

---

## 六、调用与可视化

### 6.1 运行 Agent

```python
from QueryEngine.agent import DeepSearchAgent

agent = DeepSearchAgent()

# 异步调用（推荐）
output = await agent.research_structured("DeepSeek发布新模型 各方舆论")

# 同步调用（兼容非 async 环境）
output = agent.research_structured_sync("DeepSeek发布新模型 各方舆论")
```

### 6.2 可视化界面

```bash
streamlit run SingleEngineApp/query_agent_temp_app.py
```

五个 Tab：
1. **Stance Distribution** — 各立场占比条形图
2. **Source List** — 可按立场筛选，显示 TrustScore
3. **Opinion Clusters** — 每个立场的 LLM 聚类摘要
4. **Knowledge Gaps** — 未覆盖的信息维度
5. **Social Sentiment** — MindSpider 数据：CSSD 分数、平台分解、评论情感、时序趋势折线图、多样性告警、爬取触发通知

### 6.3 加载测试数据

```bash
mysql -u root -p capstone < tests/sample_data.sql
mysql -u root -p capstone < tests/sample_comments.sql
```

### 6.4 快速评估

```bash
python -m QueryEngine.evaluation.run_evaluation --quick
python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16
python -m QueryEngine.evaluation.run_evaluation --full
```

---

## 七、已知问题与局限

| 问题 | 严重度 | 说明 |
|------|--------|------|
| ForumEngine 协作链路断开 | 高 | `research_structured()` 日志前缀不匹配 ForumEngine 监控模式。修复：在返回前调用 `self._write_forum_finding(output)`。 |
| 三引擎各搜各的 | 中 | QueryEngine、MediaEngine、InsightEngine 不共享搜索结果。 |
| `structured_summary` 为空 | 低 | `QueryAgentOutput.structured_summary` 字段尚未实现。 |
| MediaEngine 多模态未利用 | 低 | 图片和视频 modal_card 被获取后直接丢弃，只处理文字。 |
| MindSpider 非必须 | — | 系统无 MindSpider 数据时完全正常运行，社媒情感优雅降级为 null。 |

---

## 八、成员分工

| 成员 | GitHub | 主要贡献 |
|------|--------|---------|
| li_yewen | li_yewen | 项目架构搭建、Query Agent v2 设计与实现、Phase 3 MindSpider 集成 |
| MIAO Mengyu | mmy0302 | Query Agent 后续优化、英文 UI、英文文档 |
| — | Crazyheartedddd | MediaEngine LangGraph 重构、UI 英文化 |
| — | kzy1234 | app.py 集成、MediaEngine 优化、bug 修复 |
| — | Roselia-penguin | README 维护 |

---

*文档版本：v3.1 | 2026-05-02 | li_yewen*
