"""Hierarchical supervisor that fuses QueryEngine and MediaEngine evidence."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from AgentCoordinator.intelligence.contracts import (
    AgentContribution,
    AgentRunRecord,
    CoordinatorIntelligenceArtifact,
    ProviderDiagnostic,
    ResearchTask,
    ResearchTraceStep,
    RunBudget,
    utc_now,
)
from AgentCoordinator.intelligence.evidence_core import AuditKernel, EvidenceBlackboard, EvidenceCorePipeline
from AgentCoordinator.intelligence.reasoning.planner import RetrievalPlanner, task_id
from AgentCoordinator.utils.platform_profiles import canonical_social_platform

from .graph import build_fusion_graph


ProgressCallback = Callable[[str, Dict[str, Any], Dict[str, Any], float], None]
SpecialistRunner = Callable[[ResearchTask], Any]


class FusionCoordinator:
    """Parent control plane; specialists own research, EvidenceCore owns truth state."""

    def __init__(
        self,
        settings: Optional[Any] = None,
        query_runner: Optional[SpecialistRunner] = None,
        media_runner: Optional[SpecialistRunner] = None,
        use_checkpointing: bool = True,
        evaluation_hook: Optional[Callable[[CoordinatorIntelligenceArtifact], None]] = None,
    ):
        if settings is None:
            try:
                from config import settings as loaded_settings
            except Exception:
                loaded_settings = None
            settings = loaded_settings
        self.settings = settings
        self.use_checkpointing = use_checkpointing
        self.evaluation_hook = evaluation_hook
        self.planner = RetrievalPlanner()
        self.evidence_core = EvidenceCorePipeline(settings=settings)
        self.audit_kernel = AuditKernel()
        self.query_runner = query_runner or self._default_query_runner
        self.media_runner = media_runner or self._default_media_runner
        self.progress_callback: Optional[ProgressCallback] = None
        self._graph = None
        self._blackboards: Dict[str, EvidenceBlackboard] = {}
        self._pending_batches: Dict[str, Tuple[List[AgentContribution], List[AgentRunRecord]]] = {}
        self._core_results: Dict[str, Any] = {}
        self._artifacts: Dict[str, CoordinatorIntelligenceArtifact] = {}

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_fusion_graph(self)
        return self._graph

    async def run(
        self,
        query: str,
        run_id: str,
        mode: str = "query",
        max_research_rounds: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> CoordinatorIntelligenceArtifact:
        self.progress_callback = progress_callback
        max_rounds = self._max_rounds(max_research_rounds)
        initial = {
            "query": query,
            "run_id": run_id,
            "mode": mode,
            "started_at": time.time(),
            "research_round": 0,
            "max_research_rounds": max_rounds,
            "provider_diagnostics": [],
            "research_trace": [],
            "progress_state": {},
        }
        config = {"configurable": {"thread_id": run_id}} if self.use_checkpointing else None
        try:
            final_state = await self.graph.ainvoke(initial, config=config)
            if not final_state.get("artifact_ready") or run_id not in self._artifacts:
                raise RuntimeError("Fusion graph completed without a final artifact")
            return self._artifacts[run_id]
        finally:
            self._blackboards.pop(run_id, None)
            self._pending_batches.pop(run_id, None)
            self._core_results.pop(run_id, None)
            self._artifacts.pop(run_id, None)

    async def plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        understanding = self.planner.understand(state["query"])
        tasks = [
            ResearchTask(
                task_id=task_id(f"{state['run_id']}:query:breadth"),
                agent="query_agent",
                objective="Build stance-balanced breadth and discover official, supporting, opposing, and neutral sources.",
                query=understanding.query,
                task_type="breadth_and_counter_source_discovery",
                output_contract="QueryContribution",
                priority=1,
                required_stances=["official", "support", "oppose", "neutral"],
                source_scope=["web", "mindspider_db"],
                budget=RunBudget(
                    max_rounds=state["max_research_rounds"],
                    max_sources=max(
                        1,
                        int(getattr(self.settings, "COORDINATOR_QUERY_MAX_SOURCES", 120)),
                    ),
                    deadline_sec=self._task_timeout("query_agent"),
                ),
            ),
        ]
        if bool(getattr(self.settings, "COORDINATOR_ENABLE_MEDIA_AGENT", True)):
            tasks.append(ResearchTask(
                task_id=task_id(f"{state['run_id']}:media:narrative"),
                agent="media_agent",
                objective="Build section dossiers, media frames, narrative context, and multimodal source observations.",
                query=understanding.query,
                task_type="narrative_and_multimodal_depth",
                output_contract="MediaContribution",
                priority=2,
                source_scope=["web", "image", "structured_data"],
                budget=RunBudget(
                    max_rounds=state["max_research_rounds"],
                    max_sources=80,
                    deadline_sec=self._task_timeout("media_agent"),
                ),
            ))
        blackboard = EvidenceBlackboard(run_id=state["run_id"])
        blackboard.register_tasks(tasks)
        self._blackboards[state["run_id"]] = blackboard
        trace = self._trace("investigation_plan", "typed_query_media_delegation", 1, len(tasks), started)
        self._emit(state, trace, {"tasks": [task.to_dict() for task in tasks]})
        return {
            "target_entity": understanding.target_entity,
            "analysis_type": understanding.analysis_type,
            "blackboard_version": blackboard.version,
            "pending_tasks": tasks,
            "pending_contribution_count": 0,
            "provider_diagnostics": self._readiness_diagnostics(),
            "research_trace": [trace],
        }

    async def specialist_fanout_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        tasks = list(state.get("pending_tasks") or [])
        results = await asyncio.gather(*(self._execute_task(task) for task in tasks))
        contributions = [item[0] for item in results if item[0] is not None]
        runs = [item[1] for item in results]
        self._pending_batches[state["run_id"]] = (contributions, runs)
        diagnostics = list(state.get("provider_diagnostics") or [])
        for contribution, run in results:
            diagnostics.append(
                ProviderDiagnostic(
                    provider=run.agent,
                    capability="specialist_research",
                    status=("used" if run.status == "complete" else run.status),
                    route=run.task_id,
                    configured=True,
                    errors=list(run.errors),
                    metadata={"items": run.source_count, "elapsed_ms": run.elapsed_ms},
                )
            )
            if contribution is not None:
                provider_counts = Counter(item.provider for item in contribution.acquisitions)
                for provider, count in provider_counts.items():
                    diagnostics.append(
                        ProviderDiagnostic(
                            provider=provider,
                            capability="source_acquisition",
                            status="used",
                            route=contribution.agent,
                            configured=True,
                            metadata={"items": count, "task_id": contribution.task_id},
                        )
                    )
        trace = self._trace("specialist_fanout", "parallel_query_media_subgraphs", len(tasks), len(contributions), started)
        self._emit(state, trace, {"completed_agents": [item.agent for item in contributions]})
        return {
            "pending_contribution_count": len(contributions),
            "pending_tasks": [],
            "provider_diagnostics": diagnostics,
            "research_trace": list(state.get("research_trace") or []) + [trace],
        }

    async def evidence_reduce_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        blackboard = self._blackboards[state["run_id"]]
        contributions, run_batch = self._pending_batches.pop(state["run_id"], ([], []))
        runs = {run.task_id: run for run in run_batch}
        contribution_task_ids = {item.task_id for item in contributions}
        for contribution in contributions:
            blackboard.ingest(contribution, runs.get(contribution.task_id))
        for task_id_key, run in runs.items():
            if task_id_key not in contribution_task_ids:
                blackboard.record_agent_run(run)
        core_result = self.evidence_core.run(
            blackboard,
            query=state["query"],
            target_entity=state["target_entity"],
        )
        self._core_results[state["run_id"]] = core_result
        diagnostics = list(state.get("provider_diagnostics") or []) + list(core_result.provider_diagnostics)
        trace = self._trace(
            "evidence_reduce",
            "single_writer_blackboard_quality_claim_ledger",
            len(contributions),
            len(core_result.graph.normalized_items),
            started,
            notes=[f"blackboard_version={blackboard.version}"],
        )
        self._emit(state, trace, {"evidence_graph_summary": core_result.graph.graph_summary()})
        return {
            "blackboard_version": blackboard.version,
            "pending_contribution_count": 0,
            "core_version": blackboard.version,
            "evidence_graph_summary": core_result.graph.graph_summary(),
            "provider_diagnostics": diagnostics,
            "research_trace": list(state.get("research_trace") or []) + [trace],
        }

    async def global_audit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        graph = self._core_results[state["run_id"]].graph
        next_round = int(state.get("research_round") or 0) + 1
        tasks: List[ResearchTask] = []
        if next_round <= int(state.get("max_research_rounds") or 0):
            tasks = self._follow_up_tasks(state, graph)
            blackboard = self._blackboards[state["run_id"]]
            blackboard.register_tasks(tasks)
        trace = self._trace(
            "global_sufficiency_audit",
            "typed_follow_up_router" if tasks else "sufficient_or_budget_exhausted",
            len(graph.claims),
            len(tasks),
            started,
            notes=[f"round={next_round}"],
        )
        self._emit(state, trace, {"follow_up_tasks": [task.to_dict() for task in tasks]})
        return {
            "pending_tasks": tasks,
            "research_round": next_round,
            "research_trace": list(state.get("research_trace") or []) + [trace],
        }

    def audit_router(self, state: Dict[str, Any]) -> str:
        return "follow_up" if state.get("pending_tasks") else "finalize"

    async def finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        core_result = self._core_results[state["run_id"]]
        graph, synthesis_markdown = self.audit_kernel.finalize(core_result.graph, core_result.freshness_summary)
        trace = self._trace("final_audit_synthesis", "evidence_bound_audit_and_projection", len(graph.claims), len(graph.insights), started)
        traces = list(state.get("research_trace") or []) + [trace]
        diagnostics = self._dedupe_diagnostics(state.get("provider_diagnostics") or [])
        audit_summary = self._audit_summary(graph)
        source_limitations = self._source_limitations(diagnostics)
        warnings = list(dict.fromkeys((core_result.quality_summary.get("quality_warnings") or []) + source_limitations))
        snapshot = self._blackboards[state["run_id"]].snapshot()
        artifact = CoordinatorIntelligenceArtifact(
            run_id=state["run_id"],
            query=state["query"],
            mode=state.get("mode", "query"),
            created_at=utc_now(),
            target_entity=state["target_entity"],
            analysis_type=state["analysis_type"],
            evidence_graph=graph,
            evidence_graph_summary=graph.graph_summary(),
            quality_summary=core_result.quality_summary,
            freshness_summary=core_result.freshness_summary,
            source_coverage=self._source_coverage(graph),
            source_coverage_limitations=source_limitations,
            provider_diagnostics=diagnostics,
            research_trace=traces,
            audit_summary=audit_summary,
            insights=graph.insights,
            analysis_warnings=warnings,
            synthesis_markdown=synthesis_markdown,
            final_report_ready=bool(graph.insights and all(insight.citation_spans for insight in graph.insights)),
            report_engine_projection={
                "coordinator_output_latest_json_supported": True,
                "runtime_mode": "query_media_evidence_fusion",
            },
            budget_summary={
                "max_research_rounds": state["max_research_rounds"],
                "research_rounds_executed": max(
                    (task.round_index for task in snapshot.research_tasks if task.round_index > 0),
                    default=0,
                ),
                "specialist_tasks": len(snapshot.research_tasks),
                "specialist_runs": len(snapshot.agent_runs),
                "acquisition_observations": len(snapshot.acquisitions),
                "llm_calls": None,
                "api_calls": None,
                "measurement_status": "specialist_call_counters_pending_server_instrumentation",
                "evaluation_status": "deferred_until_server_demonstration",
            },
        )
        if self.evaluation_hook:
            try:
                self.evaluation_hook(artifact)
            except Exception as exc:
                logger.warning("[FusionCoordinator] evaluation hook failed: {}", exc)
        self._artifacts[state["run_id"]] = artifact
        self._emit(state, trace, {"audit_summary": audit_summary, "insights": len(graph.insights)})
        return {
            "artifact_ref": f"fusion-artifact://{state['run_id']}",
            "artifact_ready": True,
            "research_trace": traces,
        }

    async def _execute_task(self, task: ResearchTask) -> Tuple[Optional[AgentContribution], AgentRunRecord]:
        started_wall = time.time()
        started_at = utc_now()
        runner = self.query_runner if task.agent == "query_agent" else self.media_runner
        timeout = min(self._task_timeout(task.agent), max(1, int(task.budget.deadline_sec)))
        contribution = None
        errors: List[str] = []
        try:
            contribution = await asyncio.wait_for(runner(task), timeout=timeout)
            self._validate_contribution(task, contribution)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("[FusionCoordinator] {} failed for task {}", task.agent, task.task_id)
            contribution = None
        elapsed_ms = int((time.time() - started_wall) * 1000)
        contribution_errors = list(contribution.errors) if contribution else []
        errors.extend(contribution_errors)
        run = AgentRunRecord(
            run_id=f"ar_{task.task_id}_{int(started_wall * 1000)}",
            task_id=task.task_id,
            agent=task.agent,
            status=(contribution.status if contribution is not None else "error"),
            started_at=started_at,
            completed_at=utc_now(),
            elapsed_ms=elapsed_ms,
            source_count=len(contribution.sources) if contribution else 0,
            errors=errors,
        )
        return contribution, run

    def _follow_up_tasks(self, state: Dict[str, Any], graph) -> List[ResearchTask]:
        tasks: List[ResearchTask] = []
        existing = {
            task.task_id
            for task in self._blackboards[state["run_id"]].snapshot().research_tasks
        }
        query_coverage = [item for item in graph.coverage_assessments if item.agent == "query_agent"]
        missing_query_dimensions = [
            dimension
            for item in query_coverage
            for dimension in item.missing_dimensions
            if str(dimension).startswith("missing_stance:")
        ]
        if not query_coverage or missing_query_dimensions:
            missing_labels = [item.split(":", 1)[1] for item in missing_query_dimensions]
            task = ResearchTask(
                task_id=task_id(
                    f"{state['run_id']}:query:coverage:{state.get('research_round', 0)}:"
                    f"{','.join(missing_labels)}"
                ),
                agent="query_agent",
                objective="Fill missing stance and source-type coverage identified by the global audit.",
                query=state["query"],
                task_type="stance_coverage_follow_up",
                output_contract="QueryContribution",
                round_index=int(state.get("research_round") or 0) + 1,
                required_stances=missing_labels or ["official", "support", "oppose", "neutral"],
                source_scope=["web", "mindspider_db"],
                priority=1,
                budget=RunBudget(
                    max_rounds=0,
                    max_tasks=1,
                    max_sources=30,
                    max_api_calls=8,
                    max_llm_calls=8,
                    deadline_sec=self._task_timeout("query_agent"),
                ),
            )
            if task.task_id not in existing:
                tasks.append(task)
                existing.add(task.task_id)
        for claim in graph.claims[:12]:
            decision = self.audit_kernel.sufficiency.evaluate(claim)
            if decision.sufficiency == "high":
                continue
            for retrieval in self.planner.follow_up_tasks(claim, decision.reason_codes):
                task = ResearchTask(
                    task_id=task_id(f"fusion:{retrieval.task_id}"),
                    agent="query_agent",
                    objective=f"Resolve claim evidence gaps: {', '.join(decision.reason_codes) or 'insufficient coverage'}.",
                    query=retrieval.query,
                    task_type="claim_evidence_follow_up",
                    output_contract="QueryContribution",
                    round_index=int(state.get("research_round") or 0) + 1,
                    target_claim_id=claim.claim_id,
                    required_stances=["official", "oppose"] if "one_sided" in decision.reason_codes else [],
                    source_scope=[retrieval.target_source],
                    priority=retrieval.priority,
                    budget=RunBudget(
                        max_rounds=0,
                        max_tasks=1,
                        max_sources=retrieval.max_results,
                        max_api_calls=2,
                        max_llm_calls=4,
                        # QueryEngine currently executes its complete subgraph for a
                        # follow-up task, not a single HTTP retrieval call.
                        deadline_sec=max(
                            retrieval.deadline_sec,
                            min(120, self._task_timeout("query_agent")),
                        ),
                    ),
                )
                if task.task_id not in existing:
                    tasks.append(task)
                    existing.add(task.task_id)
                if len(tasks) >= 2:
                    return tasks

        media_enabled = bool(getattr(self.settings, "COORDINATOR_ENABLE_MEDIA_AGENT", True))
        media_coverage = [item for item in graph.coverage_assessments if item.agent == "media_agent"]
        if media_enabled and (not media_coverage or any(item.score < 0.7 for item in media_coverage)):
            task = ResearchTask(
                task_id=task_id(f"{state['run_id']}:media:followup:{state.get('research_round', 0)}"),
                agent="media_agent",
                objective="Fill missing section dossiers and multimodal narrative context identified by the global audit.",
                query=state["query"],
                task_type="section_dossier_follow_up",
                output_contract="MediaContribution",
                round_index=int(state.get("research_round") or 0) + 1,
                priority=3,
                budget=RunBudget(max_rounds=0, max_tasks=1, max_sources=30, max_api_calls=8, max_llm_calls=12, deadline_sec=self._task_timeout("media_agent")),
            )
            if task.task_id not in existing:
                tasks.append(task)
        return tasks[:3]

    async def _default_query_runner(self, task: ResearchTask):
        from QueryEngine.agent import DeepSearchAgent

        return await DeepSearchAgent(config=self.settings).research_contribution(task)

    async def _default_media_runner(self, task: ResearchTask):
        from MediaEngine.agent import AnspireSearchAgent, DeepSearchAgent

        search_type = getattr(self.settings, "SEARCH_TOOL_TYPE", "")
        bocha_key = (
            getattr(self.settings, "BOCHA_API_KEY", None)
            or getattr(self.settings, "BOCHA_WEB_SEARCH_API_KEY", None)
        )
        anspire_key = getattr(self.settings, "ANSPIRE_API_KEY", None)
        use_anspire = search_type == "AnspireAPI" or (not bocha_key and bool(anspire_key))
        agent_class = AnspireSearchAgent if use_anspire else DeepSearchAgent
        return await agent_class(config=self.settings).research_contribution(task)

    def _task_timeout(self, agent: str) -> int:
        field = "COORDINATOR_QUERY_AGENT_TIMEOUT" if agent == "query_agent" else "COORDINATOR_MEDIA_AGENT_TIMEOUT"
        fallback = 1800 if agent == "query_agent" else 10800
        return max(1, int(getattr(self.settings, field, fallback) if self.settings else fallback))

    @staticmethod
    def _validate_contribution(task: ResearchTask, contribution: AgentContribution) -> None:
        if not isinstance(contribution, AgentContribution):
            raise TypeError(f"{task.agent} returned {type(contribution).__name__}, expected AgentContribution")
        if contribution.task_id != task.task_id or contribution.agent != task.agent:
            raise ValueError("Specialist contribution task/agent identity does not match its delegation")
        source_ids = {item.source_id for item in contribution.sources}
        dangling_observations = [item.observation_id for item in contribution.acquisitions if item.source_id not in source_ids]
        dangling_spans = [item.span_id for item in contribution.evidence_spans if item.source_id not in source_ids]
        if dangling_observations or dangling_spans:
            raise ValueError(
                "Specialist contribution contains dangling source references: "
                f"observations={dangling_observations[:3]}, spans={dangling_spans[:3]}"
            )
        if len(source_ids) > task.budget.max_sources:
            raise ValueError(
                f"Specialist contribution exceeded max_sources={task.budget.max_sources}: {len(source_ids)}"
            )

    def _max_rounds(self, override: Optional[int]) -> int:
        if override is not None:
            return max(0, int(override))
        return max(0, int(getattr(self.settings, "COORDINATOR_MAX_RESEARCH_ROUNDS", 1) if self.settings else 1))

    def _readiness_diagnostics(self) -> List[ProviderDiagnostic]:
        settings = self.settings
        media_enabled = bool(getattr(settings, "COORDINATOR_ENABLE_MEDIA_AGENT", True)) if settings else True
        query_key = bool(getattr(settings, "QUERY_ENGINE_API_KEY", None)) if settings else False
        media_key = bool(
            getattr(settings, "MEDIA_ENGINE_API_KEY", None)
            or getattr(settings, "MINDSPIDER_API_KEY", None)
        ) if settings else False
        jina_key = bool(getattr(settings, "JINA_API_KEY", None)) if settings else False
        mindspider_enabled = bool(getattr(settings, "COORDINATOR_ENABLE_MINDSPIDER_DB", False)) if settings else False
        return [
            ProviderDiagnostic(
                provider="query_agent",
                capability="specialist_llm",
                status="configured" if query_key else "not_configured",
                route=str(getattr(settings, "QUERY_ENGINE_BASE_URL", None) or "none"),
                configured=query_key,
                model=getattr(settings, "QUERY_ENGINE_MODEL_NAME", None) if settings else None,
            ),
            ProviderDiagnostic(
                provider="media_agent",
                capability="specialist_llm",
                status="disabled" if not media_enabled else ("configured" if media_key else "not_configured"),
                route=str(
                    (getattr(settings, "MEDIA_ENGINE_BASE_URL", None) if settings else None)
                    or (getattr(settings, "MINDSPIDER_BASE_URL", None) if settings else None)
                    or "none"
                ),
                configured=media_enabled and media_key,
                model=(
                    getattr(settings, "MEDIA_ENGINE_MODEL_NAME", None)
                    or getattr(settings, "MINDSPIDER_MODEL_NAME", None)
                ) if settings else None,
            ),
            ProviderDiagnostic(
                provider="jina",
                capability="embedding_and_rerank",
                status="configured" if jina_key else "not_configured",
                route="api" if jina_key else "deterministic_rules",
                configured=jina_key,
            ),
            ProviderDiagnostic(
                provider="mindspider_db",
                capability="read_only_social_acquisition",
                status="configured" if mindspider_enabled else "disabled",
                route="query_agent" if mindspider_enabled else "web_fallback",
                configured=mindspider_enabled,
                metadata={
                    "crawl_trigger_allowed": bool(
                        getattr(settings, "COORDINATOR_ALLOW_MINDSPIDER_CRAWL_TRIGGER", False)
                    ) if settings else False,
                },
            ),
        ]

    def _emit(self, state: Dict[str, Any], trace: ResearchTraceStep, update: Dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        progress_state = dict(state.get("progress_state") or {})
        progress_state.update(update)
        try:
            self.progress_callback(trace.node, update, progress_state, time.time() - state["started_at"])
        except Exception as exc:
            logger.warning("[FusionCoordinator] progress callback failed: {}", exc)

    @staticmethod
    def _trace(node: str, route: str, input_count: int, output_count: int, started: float, notes=None) -> ResearchTraceStep:
        return ResearchTraceStep(
            node=node,
            route=route,
            input_count=input_count,
            output_count=output_count,
            elapsed_ms=int((time.time() - started) * 1000),
            notes=list(notes or []),
        )

    @staticmethod
    def _audit_summary(graph) -> Dict[str, int]:
        counts = Counter(item.decision for item in graph.audit_decisions)
        return {
            "accepted": counts.get("accept", 0),
            "weakened": counts.get("weaken", 0),
            "rejected": counts.get("reject", 0),
            "needs_search_but_budget_exhausted": counts.get("needs_search", 0),
        }

    @staticmethod
    def _source_coverage(graph) -> Dict[str, Any]:
        platforms = Counter()
        domains = Counter()
        source_types = Counter()
        for item in graph.normalized_items:
            source_types[item.source_type] += 1
            platform = canonical_social_platform(item.platform)
            if platform and item.source_type in {"ugc", "comment"}:
                platforms[platform] += 1
            else:
                domains[item.platform] += 1
        coverage_by_agent = {item.agent: item.to_dict() for item in graph.coverage_assessments}
        return {
            "web_sources": sum(domains.values()),
            "platform_sources": sum(platforms.values()),
            "social_sources": sum(platforms.values()),
            "replay_fixtures": sum(1 for item in graph.normalized_items if item.source_type == "replay_fixture"),
            "source_types": dict(source_types),
            "domains": dict(domains),
            "web_domains": dict(domains),
            "platforms": dict(platforms),
            "mindspider_platform_samples": any(item.provider == "mindspider_db" for item in graph.acquisition_observations),
            "coverage_mode": "query_media_evidence_fusion",
            "specialist_coverage": coverage_by_agent,
            "coverage_limitations": ["Observable retrieved evidence only; it is not population opinion."],
        }

    @staticmethod
    def _source_limitations(diagnostics: List[ProviderDiagnostic]) -> List[str]:
        rows = [
            "The artifact represents observable query-time evidence, not population opinion.",
            "No official platform firehose is configured.",
        ]
        if any(item.status == "error" for item in diagnostics):
            rows.append("One or more specialist/provider routes failed; see provider_diagnostics.")
        if any(
            item.capability == "specialist_llm" and item.status == "not_configured"
            for item in diagnostics
        ):
            rows.append("One or more specialist LLM routes are not configured; see provider_diagnostics.")
        if any(
            item.provider == "jina" and item.status == "not_configured"
            for item in diagnostics
        ):
            rows.append("Semantic embedding/rerank is not configured; deterministic quality rules were used.")
        return rows

    @staticmethod
    def _dedupe_diagnostics(items: List[ProviderDiagnostic]) -> List[ProviderDiagnostic]:
        rows = []
        seen = set()
        for item in items:
            key = (item.provider, item.capability, item.status, item.route, tuple(item.errors))
            if key not in seen:
                seen.add(key)
                rows.append(item)
        return rows
