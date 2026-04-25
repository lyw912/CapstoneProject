"""
Query Agent LangGraph Subgraph Builder — Phase 2

Phase 2 Complete Graph Structure (with conditional edges):

  START
    → query_planner        (Stance matrix subquery generation)
    → unified_search       (Tavily + Bocha parallel search)
    → dedup_filter         (URL + MinHash content deduplication)
    → trust_scorer         (Multi-dimensional trust scoring)       ← Phase 2 New
    → stance_classify      (Hybrid stance classification)          ← Phase 2 New
    → coverage_check       (Stance coverage check)                 ← Phase 2 New
    → [coverage_router]
        ├─ "sufficient"  → output_assemble → END
        ├─ "max_reached" → output_assemble → END
        └─ "need_more"   → gap_filler               ← Phase 2 New
                            → unified_search (Loop back for supplementary search)

Phase 3 Extension Points (commented): Integrate Crawl4AI deep extraction, LLM-based StanceClassifier.
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
# Conditional Routing Functions
# ---------------------------------------------------------------------------

def coverage_router(state: QueryAgentState) -> str:
    """
    Determine the routing path after CoverageCheck:

    - "max_reached": Search iteration limit reached → Force output
    - "need_more": Missing stances and limit not exceeded → Trigger GapFiller for supplementary search
    - "sufficient": Stance coverage satisfied → Direct output
    """
    iterations = state.get("search_iterations", 0)
    max_iter = state.get("max_iterations", 3)

    # Hard limit takes priority (prevent infinite loops)
    if iterations >= max_iter:
        return "max_reached"

    # Check for missing stances
    missing = state.get("missing_stances") or []
    if missing:
        return "need_more"

    return "sufficient"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_query_agent_graph():
    """
    Build and compile the Query Agent LangGraph subgraph (Phase 2 complete version).

    Returns:
        CompiledGraph — Can call .ainvoke(state) or .invoke(state)
    """
    graph = StateGraph(QueryAgentState)

    # ------------------------------------------------------------------
    # Register Nodes
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
    # Main Flow Edges
    # ------------------------------------------------------------------
    graph.add_edge(START,            "query_planner")
    graph.add_edge("query_planner",  "unified_search")
    graph.add_edge("unified_search", "dedup_filter")
    graph.add_edge("dedup_filter",   "trust_scorer")    # Phase 2
    graph.add_edge("trust_scorer",   "stance_classify") # Phase 2
    graph.add_edge("stance_classify","coverage_check")  # Phase 2

    # ------------------------------------------------------------------
    # Conditional Edges: Coverage check results determine routing
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "coverage_check",
        coverage_router,
        {
            "sufficient":  "output_assemble",  # Coverage sufficient → Direct output
            "need_more":   "gap_filler",        # Gap exists → Supplementary search
            "max_reached": "output_assemble",   # Iteration limit exceeded → Force output
        },
    )

    # ------------------------------------------------------------------
    # Supplementary Search Loop: GapFiller-generated subqueries back to UnifiedSearch
    # ------------------------------------------------------------------
    graph.add_edge("gap_filler",     "unified_search")

    # ------------------------------------------------------------------
    # Final Output
    # ------------------------------------------------------------------
    graph.add_edge("output_assemble", END)

    # ------------------------------------------------------------------
    # Phase 3 Extension Points (commented):
    # - Crawl4AI deep extraction node (after trust_scorer, supplement full text for high-value sources)
    # - LLM-based StanceClassifier (re-evaluate cases with low confidence from rule-based version)
    # ------------------------------------------------------------------

    return graph.compile()
