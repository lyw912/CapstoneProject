"""
Query Agent LangGraph 状态定义

所有 TypedDict 数据结构，用于在 LangGraph 图节点之间传递状态。
raw_sources 使用 Annotated[..., operator.add] 支持多轮搜索自动累积。
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# 子类型定义
# ---------------------------------------------------------------------------


class SubQueryItem(TypedDict):
    """单个子查询：由 QueryPlanner 生成，携带立场和搜索源路由信息。"""

    query: str
    target_stance: Literal["official", "support", "oppose", "neutral", "background"]
    target_source: Literal["tavily", "bocha", "insight_db", "any"]
    priority: int                        # 1（最高）~ 5（最低）
    search_params: Optional[Dict]        # 如 include_domains, time_range 等


class SourceItem(TypedDict):
    """单条搜索来源，贯穿整个处理流程并逐步被丰富。"""

    source_id: str                       # UUID
    url: str
    title: str
    source_api: str                      # "tavily" / "bocha" / "insight_db"
    platform: str                        # 提取的域名，如 "reuters.com"
    snippet: str                         # 摘要（≤500字）
    full_content: Optional[str]          # 完整正文（若搜索 API 返回）
    published_at: Optional[str]          # ISO 8601 格式日期
    trust_score: float                   # 0–1，由 TrustScorer 填充
    stance_label: Optional[str]          # 由 StanceClassifier 填充
    stance_confidence: float             # 0–1
    sub_query_ref: str                   # 来自哪个子查询
    rrf_score: Optional[float]           # RRF 融合分（Phase 2 填充）


class OpinionCluster(TypedDict):
    """同一立场的观点聚类。"""

    cluster_id: str
    stance: str
    core_argument: str                   # 核心论点（1句话）
    evidence_sources: List[str]          # source_id 列表
    representative_quote: str            # 代表性引用（≤100字）
    estimated_proportion: float          # 该立场在总来源中的占比
    source_count: int


class QueryAgentOutput(TypedDict):
    """传给下游系统的结构化输出。"""

    original_query: str
    analysis_type: str
    search_iterations: int
    total_sources_found: int             # 去重前原始总数
    total_sources_kept: int              # 去重+过滤后保留数
    stance_distribution: Dict[str, float]  # {"support": 0.3, ...}
    opinion_clusters: List[OpinionCluster]
    sources: List[SourceItem]            # 按 trust_score 降序排列
    knowledge_gaps: List[str]
    coverage_score: float                # 0–1 立场覆盖度
    structured_summary: str             # LLM 生成的综合摘要
    trace_log: List[str]


# ---------------------------------------------------------------------------
# 主状态
# ---------------------------------------------------------------------------


class QueryAgentState(TypedDict):
    """LangGraph 子图的完整运行时状态。

    raw_sources 使用 operator.add reducer，多轮搜索结果会自动追加而非覆盖。
    error_log 同理。
    """

    # === 输入 ===
    original_query: str
    analysis_type: str  # "event" / "brand" / "policy" / "person" / "general"

    # === 查询规划 ===
    sub_queries: List[SubQueryItem]
    search_iterations: int
    max_iterations: int                  # 默认 3

    # === 搜索结果（多轮累积） ===
    raw_sources: Annotated[List[SourceItem], operator.add]

    # === 处理后结果 ===
    deduped_sources: List[SourceItem]
    scored_sources: List[SourceItem]
    classified_sources: List[SourceItem]

    # === 覆盖度评估 ===
    stance_coverage: Dict[str, int]      # {"support": 3, ...}
    missing_stances: List[str]
    gap_queries: List[SubQueryItem]

    # === 最终输出 ===
    query_agent_output: Optional[QueryAgentOutput]

    # === 监控 ===
    trace_log: Annotated[List[str], operator.add]
    error_log: Annotated[List[str], operator.add]
