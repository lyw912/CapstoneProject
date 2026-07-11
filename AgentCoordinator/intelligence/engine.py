"""AgentCoordinator internal intelligence layer orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from AgentCoordinator.utils.platform_profiles import canonical_social_platform

from .acquisition.source_gateway import SourceGateway
from .contracts import (
    EvidenceGraph,
    ProviderDiagnostic,
    ResearchTraceStep,
    CoordinatorIntelligenceArtifact,
    utc_now,
)
from .quality.pipeline import QualityPipeline
from .reasoning.adaptive_loop import AdaptiveResearchLoop
from .reasoning.audit import EvidenceAuditor
from .reasoning.claim_miner import ClaimMiner
from .reasoning.planner import RetrievalPlanner
from .reasoning.sufficiency import EvidenceSufficiencyEvaluator
from .reasoning.synthesis import CitationGroundedSynthesis


ProgressCallback = Callable[[str, Dict[str, Any], Dict[str, Any], float], None]


@dataclass
class CoordinatorIntelligenceRequest:
    query: str
    thread_id: Optional[str] = None
    mode: str = "query"
    max_research_rounds: Optional[int] = None


class CoordinatorIntelligenceLayer:
    """EvidenceGraph-centered substrate for AgentCoordinator analysis."""

    def __init__(self, settings: Optional[Any] = None):
        if settings is None:
            try:
                from config import settings as loaded_settings
            except Exception:
                loaded_settings = None
            settings = loaded_settings
        self.settings = settings
        self.planner = RetrievalPlanner()
        self.gateway = SourceGateway(settings=settings)
        self.quality_pipeline = QualityPipeline(settings=settings)
        self.claim_miner = ClaimMiner()
        self.sufficiency = EvidenceSufficiencyEvaluator()
        self.auditor = EvidenceAuditor(self.sufficiency)
        self.synthesizer = CitationGroundedSynthesis()

    def run(self, request: CoordinatorIntelligenceRequest, progress_callback: Optional[ProgressCallback] = None) -> CoordinatorIntelligenceArtifact:
        run_id = request.thread_id or f"coord_{uuid.uuid4().hex[:12]}"
        started = time.time()
        trace: List[ResearchTraceStep] = []
        provider_diagnostics = self._provider_readiness()
        state: Dict[str, Any] = {"query": request.query, "run_id": run_id}

        def emit(node: str, route: str, input_count: int, output_count: int, step_started: float, update: Dict[str, Any]) -> None:
            step = ResearchTraceStep(
                node=node,
                route=route,
                input_count=input_count,
                output_count=output_count,
                elapsed_ms=int((time.time() - step_started) * 1000),
            )
            trace.append(step)
            state.update(update)
            if progress_callback:
                try:
                    progress_callback(node, update, dict(state), time.time() - started)
                except Exception as exc:  # pragma: no cover - progress reporting only
                    logger.warning("[CoordinatorIntelligenceLayer] progress callback failed: {}", exc)

        step_started = time.time()
        understanding = self.planner.understand(request.query)
        emit(
            "query_understanding",
            "rules",
            1,
            1,
            step_started,
            {
                "target_entity": understanding.target_entity,
                "analysis_type": understanding.analysis_type,
                "key_terms": understanding.key_terms,
            },
        )

        step_started = time.time()
        tasks = self.planner.initial_tasks(understanding)
        if self._mindspider_enabled():
            tasks.append(self.planner.social_platform_task(understanding))
        emit(
            "retrieval_planner",
            "structured_budget",
            1,
            len(tasks),
            step_started,
            {"retrieval_tasks": [task.to_dict() for task in tasks]},
        )

        step_started = time.time()
        items, retrieval_results, source_diagnostics = self.gateway.search_many(tasks)
        provider_diagnostics.extend(self._new_provider_diagnostics(source_diagnostics, provider_diagnostics))
        graph = EvidenceGraph(normalized_items=items, retrieval_tasks=list(tasks), retrieval_results=list(retrieval_results))
        emit(
            "source_acquisition",
            "provider_gateway",
            len(tasks),
            len(items),
            step_started,
            {
                "raw_items": len(items),
                "retrieval_results": [result.to_dict() for result in retrieval_results],
                "provider_diagnostics": [diag.to_dict() for diag in provider_diagnostics],
            },
        )

        step_started = time.time()
        quality_result = self.quality_pipeline.run(items, query=understanding.query, target_entity=understanding.target_entity)
        provider_diagnostics.extend(self._new_provider_diagnostics(quality_result.provider_diagnostics, provider_diagnostics))
        graph = quality_result.graph
        graph.retrieval_tasks = list(tasks)
        graph.retrieval_results = list(retrieval_results)
        emit(
            "quality_pipeline",
            "canonical_cluster_features",
            len(items),
            len(graph.canonical_clusters),
            step_started,
            {
                "quality_summary": quality_result.quality_summary,
                "freshness_summary": quality_result.freshness_summary.to_dict(),
                "evidence_graph_summary": graph.graph_summary(),
            },
        )

        step_started = time.time()
        graph = self.claim_miner.run(graph, target_entity=understanding.target_entity)
        emit(
            "claim_miner",
            "representative_source_spans",
            len(graph.evidence_items),
            len(graph.claims),
            step_started,
            {"claims": [claim.to_dict() for claim in graph.claims[:20]]},
        )

        max_rounds = self._max_research_rounds(request)
        step_started = time.time()
        loop = AdaptiveResearchLoop(
            planner=self.planner,
            gateway=self.gateway,
            quality_pipeline=self.quality_pipeline,
            claim_miner=self.claim_miner,
            evaluator=self.sufficiency,
            max_rounds=max_rounds,
        )
        graph, loop_trace = loop.run(graph, query=understanding.query, target_entity=understanding.target_entity)
        trace.extend(loop_trace)
        provider_diagnostics.extend(self._new_provider_diagnostics(self.gateway.diagnostics, provider_diagnostics))
        preserved_tasks = list(graph.retrieval_tasks)
        preserved_results = list(graph.retrieval_results)
        quality_result = self.quality_pipeline.run(graph.normalized_items, query=understanding.query, target_entity=understanding.target_entity)
        provider_diagnostics.extend(self._new_provider_diagnostics(quality_result.provider_diagnostics, provider_diagnostics))
        graph = quality_result.graph
        graph.retrieval_tasks = preserved_tasks
        graph.retrieval_results = preserved_results
        graph = self.claim_miner.run(graph, target_entity=understanding.target_entity)
        emit(
            "adaptive_research_loop",
            "sufficiency_routing",
            len(graph.claims),
            len(graph.retrieval_results),
            step_started,
            {
                "research_rounds": max_rounds,
                "retrieval_results": [result.to_dict() for result in graph.retrieval_results],
            },
        )

        step_started = time.time()
        graph = self.auditor.run(graph)
        audit_summary = self._audit_summary(graph)
        emit(
            "evidence_audit",
            "claim_level_judge",
            len(graph.claims),
            len(graph.audit_decisions),
            step_started,
            {"audit_summary": audit_summary, "audit_decisions": [item.to_dict() for item in graph.audit_decisions[:30]]},
        )

        step_started = time.time()
        graph, synthesis_markdown = self.synthesizer.run(graph, quality_result.freshness_summary)
        emit(
            "citation_grounded_synthesis",
            "audited_claims_only",
            len(graph.audit_decisions),
            len(graph.insights),
            step_started,
            {"insights": [insight.to_dict() for insight in graph.insights]},
        )

        source_limitations = self._source_limitations(provider_diagnostics)
        warnings = list(dict.fromkeys((quality_result.quality_summary.get("quality_warnings") or []) + source_limitations))
        artifact = CoordinatorIntelligenceArtifact(
            run_id=run_id,
            query=understanding.query,
            mode=request.mode,
            created_at=utc_now(),
            target_entity=understanding.target_entity,
            analysis_type=understanding.analysis_type,
            evidence_graph=graph,
            evidence_graph_summary=graph.graph_summary(),
            quality_summary=quality_result.quality_summary,
            freshness_summary=quality_result.freshness_summary,
            source_coverage=self._source_coverage(graph, provider_diagnostics),
            source_coverage_limitations=source_limitations,
            provider_diagnostics=provider_diagnostics,
            research_trace=trace,
            audit_summary=audit_summary,
            insights=graph.insights,
            analysis_warnings=warnings,
            synthesis_markdown=synthesis_markdown,
            final_report_ready=bool(graph.insights and all(insight.citation_spans for insight in graph.insights)),
            report_engine_projection={
                "coordinator_output_latest_json_supported": True,
                "runtime_mode": "coordinator_internal_intelligence",
            },
            budget_summary={
                "max_research_rounds": max_rounds,
                "search_tasks": len(graph.retrieval_tasks),
                "llm_calls": 0,
                "external_items": len(items),
            },
        )
        logger.info(
            "[CoordinatorIntelligenceLayer] complete raw={} canonical={} claims={} insights={}",
            artifact.evidence_graph_summary.get("raw_count"),
            artifact.evidence_graph_summary.get("canonical_count"),
            artifact.evidence_graph_summary.get("claims_count"),
            len(artifact.insights),
        )
        return artifact

    def _provider_readiness(self) -> List[ProviderDiagnostic]:
        settings = self.settings
        if not settings:
            return [
                ProviderDiagnostic(
                    provider="settings",
                    capability="configuration",
                    status="not_configured",
                    route="none",
                    configured=False,
                    warnings=["The global settings object could not be loaded."],
                )
            ]
        diagnostics = []
        jina_key = getattr(settings, "JINA_API_KEY", None)
        diagnostics.append(
            ProviderDiagnostic(
                provider="jina",
                capability="embedding_and_rerank",
                status="configured" if jina_key else "not_configured",
                route=getattr(settings, "JINA_RERANK_BASE_URL", None) or getattr(settings, "JINA_EMBEDDING_BASE_URL", None) or ("api" if jina_key else "none"),
                configured=bool(jina_key),
                model=f"{getattr(settings, 'JINA_EMBEDDING_MODEL', None) or 'jina-embeddings-v5-text-small'} / {getattr(settings, 'JINA_RERANK_MODEL', None) or 'jina-reranker-v3'}",
                warnings=[] if jina_key else ["JINA_API_KEY is missing; semantic embedding and rerank use deterministic rules only."],
            )
        )
        structured_key = getattr(settings, "QUERY_ENGINE_API_KEY", None)
        structured_model = getattr(settings, "QUERY_ENGINE_MODEL_NAME", None)
        diagnostics.append(
            ProviderDiagnostic(
                provider="structured_llm",
                capability="claim_extraction_audit",
                status="configured" if structured_key else "not_configured",
                route=getattr(settings, "QUERY_ENGINE_BASE_URL", None) or "none",
                configured=bool(structured_key),
                model=structured_model,
                warnings=[] if structured_key else ["QUERY_ENGINE_API_KEY is missing; claim extraction and audit use deterministic rules."],
            )
        )
        if bool(getattr(settings, "COORDINATOR_ENABLE_MINDSPIDER_DB", False)):
            diagnostics.append(
                ProviderDiagnostic(
                    provider="mindspider_db",
                    capability="social_source_acquisition",
                    status="configured",
                    route="COORDINATOR_ENABLE_MINDSPIDER_DB=true",
                    configured=True,
                    warnings=[],
                )
            )
        return diagnostics

    def _max_research_rounds(self, request: CoordinatorIntelligenceRequest) -> int:
        if request.max_research_rounds is not None:
            return max(0, int(request.max_research_rounds))
        if self.settings is None:
            return 1
        return max(0, int(getattr(self.settings, "COORDINATOR_MAX_RESEARCH_ROUNDS", 1) or 0))

    def _mindspider_enabled(self) -> bool:
        if self.settings is None:
            return False
        return bool(getattr(self.settings, "COORDINATOR_ENABLE_MINDSPIDER_DB", False))

    @staticmethod
    def _source_coverage(graph: EvidenceGraph, diagnostics: List[ProviderDiagnostic]) -> Dict[str, Any]:
        platform_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        source_type_counts: Dict[str, int] = {}
        social_items = 0
        web_items = 0
        replay_items = 0
        for item in graph.normalized_items:
            source_type_counts[item.source_type] = source_type_counts.get(item.source_type, 0) + 1
            platform_key = canonical_social_platform(item.platform)
            if item.source_type == "replay_fixture" or item.acquisition_source == "local_fixture":
                replay_items += 1
                continue
            if item.source_type in {"ugc", "comment"} and platform_key:
                social_items += 1
                platform_counts[platform_key] = platform_counts.get(platform_key, 0) + 1
            else:
                web_items += 1
                domain_counts[item.platform] = domain_counts.get(item.platform, 0) + 1
        mindspider_used = any(
            diag.provider == "mindspider_db" and diag.status == "used" and int(diag.metadata.get("items", 0) or 0) > 0
            for diag in diagnostics
        )
        if replay_items:
            coverage_mode = "local_replay_fixture"
        elif mindspider_used:
            coverage_mode = "web_and_mindspider_platform_samples"
        elif social_items:
            coverage_mode = "web_search_with_social_platforms"
        else:
            coverage_mode = "web_search_sources"
        return {
            "web_sources": web_items,
            "platform_sources": social_items,
            "social_sources": social_items,
            "replay_fixtures": replay_items,
            "source_types": source_type_counts,
            "domains": dict(sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))),
            "web_domains": dict(sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))),
            "platforms": dict(sorted(platform_counts.items(), key=lambda item: (-item[1], item[0]))),
            "mindspider_platform_samples": mindspider_used,
            "coverage_mode": coverage_mode,
            "coverage_limitations": [
                "Observable query-time evidence only; it is not population opinion.",
                "MindSpider platform samples and external social-platform web results are labeled separately in provider diagnostics.",
                "Web domains are not treated as platforms unless they map to a known social platform.",
            ],
        }

    @staticmethod
    def _audit_summary(graph: EvidenceGraph) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for decision in graph.audit_decisions:
            counts[decision.decision] = counts.get(decision.decision, 0) + 1
        return {
            "accepted": counts.get("accept", 0),
            "weakened": counts.get("weaken", 0),
            "rejected": counts.get("reject", 0),
            "needs_search_but_budget_exhausted": counts.get("needs_search", 0),
        }

    @staticmethod
    def _source_limitations(diagnostics: List[ProviderDiagnostic]) -> List[str]:
        limitations = [
            "The artifact represents observable query-time evidence, not population opinion.",
            "No official platform firehose is configured.",
        ]
        if any(diag.provider == "local_fixture" and diag.status == "used" for diag in diagnostics):
            limitations.append("Local replay data was used because external acquisition was unavailable or empty.")
        if any(diag.status == "not_configured" for diag in diagnostics):
            limitations.append("One or more optional providers are not configured; see provider_diagnostics.")
        if any(diag.status == "error" for diag in diagnostics):
            limitations.append("One or more configured providers failed; see provider_diagnostics.")
        return list(dict.fromkeys(limitations))

    @staticmethod
    def _new_provider_diagnostics(
        candidates: List[ProviderDiagnostic],
        existing: List[ProviderDiagnostic],
    ) -> List[ProviderDiagnostic]:
        seen = {
            (
                item.provider,
                item.capability,
                item.status,
                item.route,
                tuple(item.errors),
                tuple(item.warnings),
            )
            for item in existing
        }
        fresh = []
        for item in candidates:
            key = (
                item.provider,
                item.capability,
                item.status,
                item.route,
                tuple(item.errors),
                tuple(item.warnings),
            )
            if key not in seen:
                fresh.append(item)
                seen.add(key)
        return fresh
