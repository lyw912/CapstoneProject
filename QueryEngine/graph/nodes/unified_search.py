"""
UnifiedSearch 节点 — 统一搜索调度

调用 UnifiedSearchDispatcher 并行执行子查询，
raw_sources 通过 operator.add reducer 自动累积到 State 中。
"""

from __future__ import annotations

import asyncio
from loguru import logger

from ...tools.search_dispatcher import UnifiedSearchDispatcher
from ..state import QueryAgentState


async def unified_search_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：统一搜索。

    首轮使用 sub_queries；后续补搜轮次使用 gap_queries（由 GapFiller 生成）。
    """
    # 决定本轮搜索的子查询集合
    gap_queries = state.get("gap_queries") or []
    if gap_queries:
        queries_to_search = gap_queries
        logger.info(f"[UnifiedSearch] 补搜模式: {len(queries_to_search)} 个缺口查询")
    else:
        queries_to_search = state.get("sub_queries", [])
        logger.info(f"[UnifiedSearch] 首轮搜索: {len(queries_to_search)} 个子查询")

    iteration = state.get("search_iterations", 0) + 1

    dispatcher = UnifiedSearchDispatcher()
    new_sources, errors = await dispatcher.dispatch(queries_to_search)

    trace = (
        f"[UnifiedSearch] 第{iteration}轮, "
        f"搜索{len(queries_to_search)}个子查询, "
        f"获得{len(new_sources)}条来源"
    )
    logger.info(trace)

    return {
        "raw_sources": new_sources,       # operator.add → 自动追加
        "search_iterations": iteration,
        "gap_queries": [],                 # 清空，避免重复搜索
        "trace_log": [trace],
        "error_log": errors,
    }
