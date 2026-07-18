import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from AgentCoordinator.fusion import FusionCoordinator
from AgentCoordinator.coordinator import AgentCoordinator
from AgentCoordinator.intelligence.contracts import (
    AcquisitionObservation,
    ClaimProposal,
    CoverageAssessment,
    EvidenceCandidate,
    EvidenceSpan,
    MediaContribution,
    QueryContribution,
    ResearchTask,
    RunBudget,
    SectionDossier,
)
from AgentCoordinator.intelligence.evidence_core import EvidenceBlackboard, EvidenceCorePipeline
from AgentCoordinator.intelligence.deliberation import DebateRunner
from AgentCoordinator.intelligence.projection import build_coordinator_output_from_artifact
from AgentCoordinator.intelligence.projection.report_engine_contract import _divergence_matrix
from AgentCoordinator.utils.report_bridge import coordinator_output_to_report_engine_inputs


def _load_query_contribution_builder():
    path = Path(__file__).resolve().parents[1] / "QueryEngine" / "contribution.py"
    spec = importlib.util.spec_from_file_location("query_contribution_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_query_contribution


build_query_contribution = _load_query_contribution_builder()


class FakeDebateRunner(DebateRunner):
    def __init__(self):
        self.calls = []

    async def invoke(self, profile, phase, system_prompt, payload):
        self.calls.append((profile.role_id, phase))
        if phase == "sealed_opening":
            claim = payload["evidence"]["claims"][0]
            span_id = payload["evidence"]["spans"][0]["span_id"]
            stance = "qualify" if "public" in profile.name.lower() else "support"
            return {
                "positions": [
                    {
                        "claim_id": claim["claim_id"],
                        "stance": stance,
                        "argument": f"{profile.name} independently evaluates the cited claim.",
                        "evidence_span_ids": [span_id],
                        "assumptions": [],
                        "uncertainties": ["sample boundary"] if stance == "qualify" else [],
                        "confidence": 0.72,
                    }
                ]
            }
        if phase == "evidence_review":
            subgraph = payload["material_claims"][0]
            position = subgraph["positions"][0]
            span_id = subgraph["evidence_spans"][0]["span_id"]
            return {
                "acts": [
                    {
                        "act_type": "challenge" if profile.role_id == "skeptic" else "qualify",
                        "target_claim_id": subgraph["claim"]["claim_id"],
                        "target_position_id": position["position_id"],
                        "content": "The claim requires counter-evidence or narrower sample wording.",
                        "evidence_span_ids": [span_id],
                        "reason_codes": ["alternative_explanation"] if profile.role_id == "skeptic" else ["sample_boundary"],
                    }
                ]
            }
        if phase == "proposer_response":
            rows = []
            for challenge in payload["challenges"]:
                subgraph = next(item for item in payload["claim_subgraphs"] if item["claim"]["claim_id"] == challenge["target_claim_id"])
                rows.append(
                    {
                        "act_type": "revise",
                        "target_claim_id": challenge["target_claim_id"],
                        "target_act_id": challenge["act_id"],
                        "content": "I narrow the claim to the observable sampled evidence.",
                        "evidence_span_ids": [subgraph["evidence_spans"][0]["span_id"]],
                        "reason_codes": ["sample_boundary"],
                        "revised_claim_text": "The sampled sources contain an official response.",
                    }
                )
            return {"responses": rows}
        if phase == "post_retrieval_reassessment":
            return {"responses": []}
        if phase == "paired_blind_adjudication":
            verdicts = []
            for subgraph in payload["claim_subgraphs"]:
                verdicts.append(
                    {
                        "claim_id": subgraph["claim"]["claim_id"],
                        "decision": "weaken",
                        "reason_codes": ["sample_boundary"],
                        "explanation": "The span supports a bounded observation, not a population claim.",
                        "required_edit": "Use sampled-source wording.",
                        "final_wording": "The sampled sources contain an official response.",
                        "decisive_act_ids": [item["act_id"] for item in subgraph["argument_acts"]],
                        "evidence_span_ids": [item["span_id"] for item in subgraph["evidence_spans"]],
                        "confidence": 0.74,
                    }
                )
            return {"verdicts": verdicts}
        raise AssertionError(f"Unexpected debate phase: {phase}")


def source(source_id, text):
    return EvidenceCandidate(
        source_id=source_id,
        platform="example.com",
        source_type="mainstream_media",
        source_name="Example News",
        source_item_id=source_id,
        url="https://example.com/story?utm_source=test",
        canonical_url="https://example.com/story",
        title="Official response and public reaction",
        text=text,
        language="en",
        published_at="2026-07-10T12:00:00Z",
    )


def observation(observation_id, source_id, task_id, agent, query):
    return AcquisitionObservation(
        observation_id=observation_id,
        source_id=source_id,
        task_id=task_id,
        agent=agent,
        query=query,
        provider="fake_provider",
        tool="fake_search",
        observed_at="2026-07-11T12:00:00Z",
        retrieved_at="2026-07-11T12:00:00Z",
        rank=1,
    )


def fake_query(task):
    source_id = f"query_{task.task_id}"
    span_id = f"span_{task.task_id}"
    return QueryContribution(
        contribution_id=f"query_contribution_{task.task_id}",
        task_id=task.task_id,
        agent="query_agent",
        status="complete",
        sources=[source(source_id, "The official response confirms an update while public reaction remains mixed.")],
        acquisitions=[observation("query_obs", source_id, task.task_id, "query_agent", task.query)],
        evidence_spans=[
            EvidenceSpan(
                span_id=span_id,
                source_id=source_id,
                text="The official response confirms an update.",
                start_char=0,
                end_char=41,
                span_type="source_excerpt",
                confidence=0.8,
            )
        ],
        claim_proposals=[
            ClaimProposal(
                proposal_id=f"proposal_{task.task_id}",
                agent="query_agent",
                claim_text="The organization published an official response.",
                claim_type="fact",
                target_entity="Example",
                aspect="official_response",
                stance="official",
                evidence_span_ids=[span_id],
                task_id=task.task_id,
                confidence=0.8,
            )
        ],
        coverage=CoverageAssessment(
            assessment_id=f"coverage_{task.task_id}",
            task_id=task.task_id,
            agent="query_agent",
            score=1.0,
            stance_counts={"official": 1, "neutral": 1, "support": 1, "oppose": 1},
        ),
        stance_distribution={"official": 0.25, "neutral": 0.25, "support": 0.25, "oppose": 0.25},
    )


def fake_media(task):
    source_id = f"media_{task.task_id}"
    span_id = f"media_span_{task.task_id}"
    summary = "Coverage frames the response as important but notes unresolved public questions."
    return MediaContribution(
        contribution_id=f"media_contribution_{task.task_id}",
        task_id=task.task_id,
        agent="media_agent",
        status="complete",
        sources=[source(source_id, summary + " Additional narrative and timeline context." )],
        acquisitions=[observation("media_obs", source_id, task.task_id, "media_agent", task.query)],
        evidence_spans=[
            EvidenceSpan(
                span_id=span_id,
                source_id=source_id,
                text=summary,
                start_char=0,
                end_char=len(summary),
                span_type="media_excerpt",
                confidence=0.75,
            )
        ],
        coverage=CoverageAssessment(
            assessment_id=f"media_coverage_{task.task_id}",
            task_id=task.task_id,
            agent="media_agent",
            score=1.0,
            covered_dimensions=["Media framing"],
        ),
        dossiers=[
            SectionDossier(
                dossier_id=f"dossier_{task.task_id}",
                task_id=task.task_id,
                section_id="media_framing",
                title="Media framing",
                objective="Explain narrative frames.",
                summary=summary,
                source_ids=[source_id],
                evidence_span_ids=[span_id],
                multimodal_assets=[{"type": "image", "url": "https://example.com/image.jpg"}],
                reflection_rounds=1,
            )
        ],
        narrative_summary=summary,
    )


class EvidenceBlackboardContractTestCase(unittest.TestCase):
    def test_divergence_uses_channel_groups_and_excludes_tiny_samples(self):
        items = []
        clusters = []
        quality = {}

        def add(item_id, platform, source_type, stance):
            items.append(SimpleNamespace(item_id=item_id, platform=platform, source_type=source_type))
            clusters.append(
                SimpleNamespace(
                    representative_item_id=item_id,
                    member_item_ids=[item_id],
                )
            )
            sentiment = "positive" if stance == "support" else "negative" if stance == "oppose" else "neutral"
            quality[item_id] = SimpleNamespace(stance=stance, sentiment=sentiment)

        add("official_1", "agency-a.gov", "official", "official")
        add("official_2", "agency-b.gov", "official", "official")
        add("official_3", "agency-c.gov", "official", "neutral")
        add("web_1", "news-a.example", "search_result", "support")
        add("web_2", "news-b.example", "search_result", "neutral")
        add("web_3", "news-c.example", "search_result", "oppose")
        items.append(SimpleNamespace(item_id="web_1_counter", platform="news-a.example", source_type="search_result"))
        clusters[3].member_item_ids.append("web_1_counter")
        quality["web_1_counter"] = SimpleNamespace(stance="oppose", sentiment="negative")
        add("weibo_1", "weibo", "ugc", "support")
        add("weibo_2", "weibo", "ugc", "support")
        add("weibo_3", "weibo", "ugc", "oppose")
        add("twitter_1", "twitter", "ugc", "oppose")

        graph = SimpleNamespace(canonical_clusters=clusters, item_index=lambda: {item.item_id: item for item in items})
        result = _divergence_matrix(graph, quality)

        self.assertEqual(result["group_counts"], {"official_web": 3, "web_media": 3, "weibo": 3})
        self.assertEqual(result["excluded_low_sample_groups"], {"twitter": 1})
        self.assertNotIn("agency-a.gov", " ".join(result["pairs"]))
        self.assertTrue(result["pairs"])
        self.assertTrue(all(0.0 <= value < 1.0 for value in result["pairs"].values()))
        self.assertEqual(
            result["group_distributions"]["web_media"],
            {"neutral": 0.3333, "oppose": 0.4444, "support": 0.2222},
        )
        self.assertTrue(
            all(
                set(distribution) <= {"support", "neutral", "oppose"}
                for distribution in result["group_distributions"].values()
            )
        )

    def test_query_contribution_promotes_mindspider_voices_to_evidence(self):
        task = ResearchTask(
            task_id="query_social",
            agent="query_agent",
            objective="Collect public voices",
            query="DeepSeek API pricing",
            task_type="breadth",
            output_contract="QueryContribution",
            budget=RunBudget(max_sources=10),
        )
        contribution = build_query_contribution(
            {
                "sources": [],
                "coverage_score": 0.5,
                "social_sentiment": {
                    "mode": "available",
                    "evidence_posts": [
                        {
                            "platform": "weibo",
                            "content": "Users are comparing the new API price with competitors.",
                            "url": "mindspider://weibo/weibo_note/abc123",
                            "publish_time": "2026-07-11T08:00:00",
                            "stance": "neutral",
                        }
                    ],
                },
            },
            task,
        )

        self.assertEqual(len(contribution.sources), 1)
        self.assertEqual(contribution.sources[0].source_type, "ugc")
        self.assertEqual(contribution.acquisitions[0].provider, "mindspider_db")
        self.assertEqual(contribution.evidence_spans[0].span_type, "social_voice_excerpt")

    def test_same_source_is_canonicalized_but_acquisitions_are_preserved(self):
        board = EvidenceBlackboard(run_id="fusion_test")
        query_contribution = fake_query(SimpleNamespace(task_id="query_task", query="example"))
        media_contribution = fake_media(SimpleNamespace(task_id="media_task", query="example"))

        board.ingest(query_contribution)
        board.ingest(media_contribution)
        snapshot = board.snapshot()

        self.assertEqual(len(snapshot.sources), 1)
        self.assertEqual(len(snapshot.acquisitions), 2)
        self.assertEqual({item.agent for item in snapshot.acquisitions}, {"query_agent", "media_agent"})
        self.assertGreater(len(snapshot.sources[0].text), len(query_contribution.sources[0].text))

    def test_claim_proposal_enters_ledger_only_with_bound_source_span(self):
        board = EvidenceBlackboard(run_id="claim_test")
        board.ingest(fake_query(SimpleNamespace(task_id="query_task", query="example")))
        result = EvidenceCorePipeline(settings=None).run(board, query="example", target_entity="Example")

        proposed = [claim for claim in result.graph.claims if claim.created_by.startswith("query_agent")]
        self.assertEqual(len(proposed), 1)
        self.assertTrue(proposed[0].supporting_spans)
        self.assertTrue(result.graph.support_edges)

    def test_run_sync_does_not_retry_operational_runtime_error(self):
        coordinator = AgentCoordinator(use_checkpointing=False)
        calls = []

        async def fail_once(*_args, **_kwargs):
            calls.append("called")
            raise RuntimeError("operational failure")

        coordinator.run = fail_once
        with self.assertRaisesRegex(RuntimeError, "operational failure"):
            coordinator.run_sync("example")
        self.assertEqual(calls, ["called"])


class FusionCoordinatorContractTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_dual_chamber_executes_independent_roles_and_paired_judges(self):
        settings = SimpleNamespace(
            JINA_API_KEY=None,
            COORDINATOR_ENABLE_DEBATE=True,
            COORDINATOR_DEBATE_MAX_MATERIAL_CLAIMS=6,
            COORDINATOR_DEBATE_MAX_LLM_CALLS=18,
            COORDINATOR_DEBATE_TIMEOUT=30,
            COORDINATOR_DEBATE_SCHEMA_RETRIES=0,
            COORDINATOR_DEBATE_ROLE_ROUTES='{"*":"query"}',
            COORDINATOR_ENABLE_MEDIA_AGENT=True,
            COORDINATOR_MAX_EMBEDDING_ITEMS=0,
            COORDINATOR_MAX_RERANK_DOCUMENTS=0,
            COORDINATOR_QUERY_AGENT_TIMEOUT=5,
            COORDINATOR_MEDIA_AGENT_TIMEOUT=5,
            COORDINATOR_MAX_RESEARCH_ROUNDS=0,
        )

        async def query_runner(task):
            return fake_query(task)

        async def media_runner(task):
            return fake_media(task)

        debate_runner = FakeDebateRunner()
        coordinator = FusionCoordinator(
            settings=settings,
            query_runner=query_runner,
            media_runner=media_runner,
            debate_runner=debate_runner,
            use_checkpointing=False,
        )
        artifact = await coordinator.run(
            query="Example public response",
            run_id="dual_chamber_contract",
            max_research_rounds=0,
        )

        session = artifact.debate_session
        self.assertIsNotNone(session)
        self.assertEqual(len(session.positions), 4)
        self.assertEqual(len({item.agent_id for item in session.positions}), 4)
        self.assertEqual({item.actor_id for item in session.argument_acts if item.actor_id in {"skeptic", "methodologist"}}, {"skeptic", "methodologist"})
        self.assertTrue(session.revisions)
        self.assertEqual({item.judge_id for item in session.verdicts}, {"primary_judge", "review_judge"})
        self.assertEqual(session.independence_summary["configured_mode"], "same_model_fallback")
        self.assertTrue(session.output_groups["audited_findings"])
        self.assertGreaterEqual(artifact.budget_summary["debate_llm_calls"], 9)
        self.assertEqual(artifact.report_engine_projection["deliberation_mode"], "dual_chamber_evidence_debate")
        output = build_coordinator_output_from_artifact(artifact, duration_seconds=0.1)
        report_inputs = coordinator_output_to_report_engine_inputs(output)
        self.assertEqual(output["investigation_brief"]["original_query"], "Example public response")
        self.assertEqual(output["deliberation"]["status"], session.status)
        self.assertIn("[Audited Findings]", report_inputs["forum_logs"])
        self.assertIn("[Investigation Brief]", report_inputs["forum_logs"])
        self.assertIn("[Binding Evidence Policy]", report_inputs["forum_logs"])
        self.assertIn("Binding Evidence Policy", report_inputs["reports"][0])
        self.assertNotIn("Structured Coordinator Output", report_inputs["reports"][0])
        self.assertLessEqual(len(report_inputs["reports"][0]), 30000)
        opening_agents = {agent for agent, phase in debate_runner.calls if phase == "sealed_opening"}
        self.assertEqual(len(opening_agents), 4)

    async def test_media_specialist_can_be_explicitly_disabled(self):
        settings = SimpleNamespace(
            JINA_API_KEY=None,
            COORDINATOR_ENABLE_MEDIA_AGENT=False,
            COORDINATOR_MAX_EMBEDDING_ITEMS=0,
            COORDINATOR_MAX_RERANK_DOCUMENTS=0,
            COORDINATOR_QUERY_AGENT_TIMEOUT=5,
            COORDINATOR_MEDIA_AGENT_TIMEOUT=5,
            COORDINATOR_MAX_RESEARCH_ROUNDS=0,
        )

        async def media_runner(_task):
            raise AssertionError("Media runner must not be called")

        async def query_runner(task):
            return fake_query(task)

        coordinator = FusionCoordinator(
            settings=settings,
            query_runner=query_runner,
            media_runner=media_runner,
            use_checkpointing=False,
        )
        artifact = await coordinator.run("Example public response", "query_only", max_research_rounds=0)

        self.assertEqual(artifact.evidence_graph_summary["specialist_contributions"], 1)
        media_diagnostic = next(
            item for item in artifact.provider_diagnostics if item.provider == "media_agent"
        )
        self.assertEqual(media_diagnostic.status, "disabled")

    async def test_parent_graph_keeps_external_artifact_and_report_contract(self):
        settings = SimpleNamespace(
            JINA_API_KEY=None,
            COORDINATOR_MAX_EMBEDDING_ITEMS=0,
            COORDINATOR_MAX_RERANK_DOCUMENTS=0,
            COORDINATOR_QUERY_AGENT_TIMEOUT=5,
            COORDINATOR_MEDIA_AGENT_TIMEOUT=5,
            COORDINATOR_MAX_RESEARCH_ROUNDS=0,
        )

        async def query_runner(task):
            return fake_query(task)

        async def media_runner(task):
            return fake_media(task)

        coordinator = FusionCoordinator(
            settings=settings,
            query_runner=query_runner,
            media_runner=media_runner,
            use_checkpointing=False,
        )
        artifact = await coordinator.run(
            query="Example public response",
            run_id="fusion_contract",
            max_research_rounds=0,
        )
        output = build_coordinator_output_from_artifact(artifact, duration_seconds=0.1)
        report_inputs = coordinator_output_to_report_engine_inputs(output)

        self.assertEqual(output["schema_version"], "2.1-coordinator-intelligence")
        self.assertEqual(output["artifact_derivation"]["primary_record"], "coordinator_intelligence")
        self.assertTrue(output["source_data"]["media_agent"]["available"])
        self.assertEqual(output["source_data"]["media_agent"]["section_dossiers"], 1)
        self.assertIn("Media Section Dossiers", report_inputs["reports"][1])
        self.assertGreaterEqual(artifact.evidence_graph_summary["acquisition_observations"], 2)
        self.assertEqual(artifact.report_engine_projection["runtime_mode"], "query_media_evidence_fusion")
        self.assertEqual(artifact.budget_summary["evaluation_status"], "engineering_acceptance_only")


if __name__ == "__main__":
    unittest.main()
