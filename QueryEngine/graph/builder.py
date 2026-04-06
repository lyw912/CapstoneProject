"""
Query Agent LangGraph 子图构建 — Phase 2

Phase 2 完整图结构（含条件边）：

  START
    → query_planner        （立场矩阵子查询生成）
    → unified_search       （Tavily + Bocha 并行搜索）
    → dedup_filter         （URL + MinHash 内容去重）
    → trust_scorer         （多维可信度评分）       ← Phase 2 新增
    → stance_classify      （混合立场分类）          ← Phase 2 新增
    → coverage_check       （立场覆盖度检查）        ← Phase 2 新增
    → [coverage_router]
        ├─ "sufficient"  → output_assemble → END
        ├─ "max_reached" → output_assemble → END
        └─ "need_more"   → gap_filler               ← Phase 2 新增
                            → unified_search （循环补搜）

Phase 3 扩展点（已注释）：接入 Crawl4AI 深度提取、LLM 版 StanceClassifier。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    coverage_check_node,
    dedup_filter_node,
    gap_filler_node,
    output_assemble_node,
    query_planner_node,
    stance_classify_node,
    trust_scorer_node,
    unified_search_node,
)
from .state import QueryAgentState


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------

def coverage_router(state: QueryAgentState) -> str:
    """
    决定 CoverageCheck 之后的走向：

    - "max_reached"：已达搜索轮次上限 → 强制输出
    - "need_more"  ：有缺失立场且未超上限 → 触发 GapFiller 补搜
    - "sufficient" ：立场覆盖已满足 → 直接输出
    """
    iterations = state.get("search_iterations", 0)
    max_iter = state.get("max_iterations", 3)

    # 硬性上限优先（防止无限循环）
    if iterations >= max_iter:
        return "max_reached"

    # 检查缺失立场
    missing = state.get("missing_stances") or []
    if missing:
        return "need_more"

    return "sufficient"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_query_agent_graph():
    """
    构建并编译 Query Agent LangGraph 子图（Phase 2 完整版）。

    Returns:
        CompiledGraph — 可调用 .ainvoke(state) 或 .invoke(state)
    """
    graph = StateGraph(QueryAgentState)

    # ------------------------------------------------------------------
    # 注册节点
    # ------------------------------------------------------------------
    graph.add_node("query_planner",   query_planner_node)
    graph.add_node("unified_search",  unified_search_node)
    graph.add_node("dedup_filter",    dedup_filter_node)
    graph.add_node("trust_scorer",    trust_scorer_node)    # Phase 2
    graph.add_node("stance_classify", stance_classify_node) # Phase 2
    graph.add_node("coverage_check",  coverage_check_node)  # Phase 2
    graph.add_node("gap_filler",      gap_filler_node)      # Phase 2
    graph.add_node("output_assemble", output_assemble_node)

    # ------------------------------------------------------------------
    # 主流程边
    # ------------------------------------------------------------------
    graph.add_edge(START,            "query_planner")
    graph.add_edge("query_planner",  "unified_search")
    graph.add_edge("unified_search", "dedup_filter")
    graph.add_edge("dedup_filter",   "trust_scorer")    # Phase 2
    graph.add_edge("trust_scorer",   "stance_classify") # Phase 2
    graph.add_edge("stance_classify","coverage_check")  # Phase 2

    # ------------------------------------------------------------------
    # 条件边：覆盖度检查结果决定走向
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "coverage_check",
        coverage_router,
        {
            "sufficient":  "output_assemble",  # 覆盖度满足 → 直接输出
            "need_more":   "gap_filler",        # 有缺口 → 补搜
            "max_reached": "output_assemble",   # 超轮次上限 → 强制输出
        },
    )

    # ------------------------------------------------------------------
    # 补搜回路：GapFiller 生成的子查询回到 UnifiedSearch
    # ------------------------------------------------------------------
    graph.add_edge("gap_filler",     "unified_search")

    # ------------------------------------------------------------------
    # 最终输出
    # ------------------------------------------------------------------
    graph.add_edge("output_assemble", END)

    # ------------------------------------------------------------------
    # Phase 3 扩展点（已注释）：
    # - Crawl4AI 深度提取节点（在 trust_scorer 之后，对高价值来源补充全文）
    # - LLM 版 StanceClassifier（对规则版置信度低的 case 二次确认）
    # ------------------------------------------------------------------

    return graph.compile()
