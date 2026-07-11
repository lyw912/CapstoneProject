"""Build the parent fusion LangGraph with a bounded global audit loop."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .state import FusionState


def build_fusion_graph(supervisor):
    graph = StateGraph(FusionState)
    graph.add_node("investigation_plan", supervisor.plan_node)
    graph.add_node("specialist_fanout", supervisor.specialist_fanout_node)
    graph.add_node("evidence_reduce", supervisor.evidence_reduce_node)
    graph.add_node("global_sufficiency_audit", supervisor.global_audit_node)
    graph.add_node("final_audit_synthesis", supervisor.finalize_node)

    graph.add_edge(START, "investigation_plan")
    graph.add_edge("investigation_plan", "specialist_fanout")
    graph.add_edge("specialist_fanout", "evidence_reduce")
    graph.add_edge("evidence_reduce", "global_sufficiency_audit")
    graph.add_conditional_edges(
        "global_sufficiency_audit",
        supervisor.audit_router,
        {"follow_up": "specialist_fanout", "finalize": "final_audit_synthesis"},
    )
    graph.add_edge("final_audit_synthesis", END)
    if supervisor.use_checkpointing:
        from langgraph.checkpoint.memory import MemorySaver

        return graph.compile(checkpointer=MemorySaver())
    return graph.compile()
