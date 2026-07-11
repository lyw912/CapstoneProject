"""Adaptive research loop driven by claim sufficiency."""

from __future__ import annotations

import time
from typing import List, Tuple

from ..acquisition.source_gateway import SourceGateway
from ..contracts import EvidenceGraph, ResearchTraceStep, RetrievalTask
from ..quality.pipeline import QualityPipeline
from .claim_miner import ClaimMiner
from .planner import RetrievalPlanner
from .sufficiency import EvidenceSufficiencyEvaluator


class AdaptiveResearchLoop:
    """Run bounded follow-up retrieval for insufficient or one-sided claims."""

    def __init__(
        self,
        planner: RetrievalPlanner,
        gateway: SourceGateway,
        quality_pipeline: QualityPipeline,
        claim_miner: ClaimMiner,
        evaluator: EvidenceSufficiencyEvaluator,
        max_rounds: int = 1,
    ):
        self.planner = planner
        self.gateway = gateway
        self.quality_pipeline = quality_pipeline
        self.claim_miner = claim_miner
        self.evaluator = evaluator
        self.max_rounds = max(0, int(max_rounds))

    def run(self, graph: EvidenceGraph, query: str, target_entity: str) -> Tuple[EvidenceGraph, List[ResearchTraceStep]]:
        trace: List[ResearchTraceStep] = []
        if self.max_rounds <= 0:
            return graph, trace

        for round_index in range(1, self.max_rounds + 1):
            started = time.time()
            tasks = self._tasks_for_graph(graph)
            if not tasks:
                trace.append(
                    ResearchTraceStep(
                        node="adaptive_research_loop",
                        route="sufficient_or_no_follow_up",
                        input_count=len(graph.claims),
                        output_count=0,
                        elapsed_ms=int((time.time() - started) * 1000),
                        notes=[f"round={round_index}"],
                    )
                )
                break

            graph.retrieval_tasks.extend(tasks)
            new_items, results, _diagnostics = self.gateway.search_many(tasks)
            graph.retrieval_results.extend(results)
            if not new_items:
                trace.append(
                    ResearchTraceStep(
                        node="adaptive_research_loop",
                        route="no_new_evidence",
                        input_count=len(tasks),
                        output_count=0,
                        elapsed_ms=int((time.time() - started) * 1000),
                        notes=[f"round={round_index}"],
                    )
                )
                break

            pipeline_result = self.quality_pipeline.merge(graph, new_items, query=query, target_entity=target_entity)
            graph = self.claim_miner.run(pipeline_result.graph, target_entity=target_entity)
            graph.retrieval_tasks.extend(tasks)
            graph.retrieval_results.extend(results)
            trace.append(
                ResearchTraceStep(
                    node="adaptive_research_loop",
                    route="claim_driven_follow_up",
                    input_count=len(tasks),
                    output_count=len(new_items),
                    elapsed_ms=int((time.time() - started) * 1000),
                    notes=[f"round={round_index}"],
                )
            )
        return graph, trace

    def _tasks_for_graph(self, graph: EvidenceGraph) -> List[RetrievalTask]:
        tasks: List[RetrievalTask] = []
        existing = {task.task_id for task in graph.retrieval_tasks}
        for claim in graph.claims[:12]:
            decision = self.evaluator.evaluate(claim)
            if decision.sufficiency == "high":
                continue
            for task in self.planner.follow_up_tasks(claim, decision.reason_codes):
                if task.task_id not in existing:
                    tasks.append(task)
                    existing.add(task.task_id)
            if len(tasks) >= 4:
                break
        return tasks
