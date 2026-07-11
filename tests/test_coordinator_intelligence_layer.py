import unittest
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from AgentCoordinator.intelligence.acquisition.source_gateway import SourceGateway
from AgentCoordinator.intelligence.contracts import NormalizedItem, RetrievalTask
from AgentCoordinator.intelligence import CoordinatorIntelligenceLayer, CoordinatorIntelligenceRequest
from AgentCoordinator.intelligence.quality.pipeline import QualityPipeline
from AgentCoordinator.intelligence.projection import build_coordinator_output_from_artifact
from AgentCoordinator.intelligence.reasoning.planner import RetrievalPlanner


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeMindSpiderRow:
    platform = "weibo"
    source_table = "weibo_note"
    title_or_content = "Users discuss DeepSeek support delays on Weibo with specific ticket examples."
    url = "https://weibo.example/status/1"
    publish_time = datetime(2026, 7, 8, 10, 0, 0)
    hotness_score = None
    source_keyword = "DeepSeek"


class FakeMindSpiderResponse:
    results = [FakeMindSpiderRow()]
    total = 1


class FakeMindSpiderDB:
    @property
    def engine(self):
        return self

    @contextmanager
    def connect(self):
        yield self

    def search_topic_globally(self, _query, limit_per_table=20):
        return FakeMindSpiderResponse()


class CoordinatorIntelligenceLayerTestCase(unittest.TestCase):
    def test_planner_expands_english_brief_for_chinese_platform_search(self):
        planner = RetrievalPlanner()
        understanding = planner.understand("Public reaction to rising prices and cost-of-living pressure in China")
        social_task = planner.social_platform_task(understanding)
        web_tasks = planner.initial_tasks(understanding)

        self.assertIn("物价上涨", social_task.query_variants)
        self.assertIn("生活成本", social_task.query_variants)
        self.assertIn("通胀", social_task.query_variants)
        self.assertEqual(social_task.query, "物价上涨")
        self.assertTrue(any("public reaction" in query.lower() for query in web_tasks[0].query_variants))
        self.assertTrue(any("rising price" in query.lower() or "cost of living" in query.lower() for query in web_tasks[0].query_variants))

    def test_local_replay_produces_audited_cited_artifact(self):
        engine = CoordinatorIntelligenceLayer()
        if engine.gateway.settings is not None:
            engine.gateway.settings.ANSPIRE_API_KEY = None
            engine.gateway.settings.BOCHA_WEB_SEARCH_API_KEY = None
            engine.gateway.settings.TAVILY_API_KEY = None
            engine.gateway.settings.JINA_API_KEY = None
            engine.gateway.settings.COORDINATOR_ALLOW_REPLAY_FALLBACK = True

        artifact = engine.run(
            CoordinatorIntelligenceRequest(
                query="DeepSeek customer support controversy",
                thread_id="unit_signal",
                max_research_rounds=1,
            )
        )
        output = build_coordinator_output_from_artifact(artifact, duration_seconds=0.5)

        self.assertEqual(output["schema_version"], "2.1-coordinator-intelligence")
        self.assertIn("coordinator_intelligence", output)
        self.assertEqual(output["artifact_derivation"]["primary_record"], "coordinator_intelligence")
        self.assertGreater(artifact.evidence_graph_summary["raw_count"], 0)
        self.assertGreater(artifact.evidence_graph_summary["canonical_count"], 0)
        self.assertLess(
            artifact.evidence_graph_summary["canonical_count"],
            artifact.evidence_graph_summary["raw_count"],
        )
        self.assertTrue(artifact.final_report_ready)
        self.assertTrue(artifact.insights)
        self.assertTrue(all(insight.citation_spans for insight in artifact.insights))
        self.assertTrue(
            any(
                diagnostic.provider == "local_fixture" and diagnostic.status == "used"
                for diagnostic in artifact.provider_diagnostics
            )
        )
        self.assertTrue(
            any(
                diagnostic.provider == "jina" and diagnostic.status == "not_configured"
                for diagnostic in artifact.provider_diagnostics
            )
        )
        self.assertIn(
            "Repeated posts are treated as coverage strength, not independent viewpoints.",
            artifact.quality_summary["quality_warnings"],
        )

    def test_jina_semantic_route_updates_quality_features(self):
        settings = SimpleNamespace(
            JINA_API_KEY="test-key",
            JINA_EMBEDDING_BASE_URL="https://api.jina.ai/v1/embeddings",
            JINA_EMBEDDING_MODEL="jina-embeddings-v5-text-small",
            JINA_EMBEDDING_DIMENSIONS="",
            JINA_RERANK_BASE_URL="https://api.jina.ai/v1/rerank",
            JINA_RERANK_MODEL="jina-reranker-v3",
            COORDINATOR_MAX_EMBEDDING_ITEMS=20,
            COORDINATOR_MAX_RERANK_DOCUMENTS=20,
            COORDINATOR_PROVIDER_TIMEOUT=5,
            COORDINATOR_SEMANTIC_DUPLICATE_THRESHOLD=0.90,
        )
        items = [
            NormalizedItem(
                item_id="item_1",
                raw_id="raw_1",
                platform="news",
                source_type="mainstream_media",
                source_name="Example News",
                source_item_id="raw_1",
                url="https://example.com/a",
                canonical_url="https://example.com/a",
                author_id_hash=None,
                title="DeepSeek support update",
                text="DeepSeek customer support tickets were delayed according to a detailed support update.",
                language="en",
                published_at="2026-07-08T10:00:00Z",
                observed_at="2026-07-08T10:01:00Z",
                retrieved_at="2026-07-08T10:02:00Z",
                retrieval_query="DeepSeek support",
                raw_ref="test://1",
            ),
            NormalizedItem(
                item_id="item_2",
                raw_id="raw_2",
                platform="reddit",
                source_type="ugc",
                source_name="Forum",
                source_item_id="raw_2",
                url="https://example.com/b",
                canonical_url="https://example.com/b",
                author_id_hash=None,
                title="Support counterpoint",
                text="Some users say DeepSeek support resolved their tickets after escalation.",
                language="en",
                published_at="2026-07-08T10:05:00Z",
                observed_at="2026-07-08T10:06:00Z",
                retrieved_at="2026-07-08T10:07:00Z",
                retrieval_query="DeepSeek support",
                raw_ref="test://2",
            ),
        ]

        def fake_post(url, **kwargs):
            if url.endswith("/embeddings"):
                self.assertEqual(kwargs["json"]["task"], "clustering")
                return FakeResponse({"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]})
            if url.endswith("/rerank"):
                return FakeResponse({"results": [{"index": 0, "relevance_score": 0.91}, {"index": 1, "relevance_score": 0.42}]})
            raise AssertionError(url)

        with patch("AgentCoordinator.intelligence.providers.semantic.requests.post", side_effect=fake_post):
            result = QualityPipeline(settings=settings).run(items, query="DeepSeek support", target_entity="DeepSeek")

        self.assertTrue(any(item.provider == "jina" and item.capability == "embedding" and item.status == "used" for item in result.provider_diagnostics))
        self.assertTrue(any(item.provider == "jina" and item.capability == "rerank" and item.status == "used" for item in result.provider_diagnostics))
        features = {feature.item_id: feature for feature in result.graph.quality_features}
        self.assertEqual(features["item_1"].judge_route, "jina_rerank")
        self.assertEqual(features["item_1"].relevance_score, 0.91)

    def test_search_tool_type_selects_tavily_without_anspire_fallback(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY="tavily-key",
            ANSPIRE_API_KEY="anspire-key",
            BOCHA_WEB_SEARCH_API_KEY=None,
            SEARCH_TIMEOUT=5,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=False,
        )
        task = RetrievalTask(
            task_id="task_1",
            parent_claim_id=None,
            query="DeepSeek support update",
            query_variants=[],
            target_source="web",
            purpose="source_acquisition",
            priority=1,
            deadline_sec=30,
            max_results=3,
            budget={},
            created_by="unit_test",
        )

        def fake_post(url, **_kwargs):
            self.assertEqual(url, "https://api.tavily.com/search")
            return FakeResponse(
                {
                    "results": [
                        {
                            "url": "https://example.com/tavily",
                            "title": "Tavily result",
                            "content": "Tavily returned the configured source acquisition result.",
                        }
                    ]
                }
            )

        with patch("AgentCoordinator.intelligence.acquisition.source_gateway.requests.post", side_effect=fake_post):
            items, results, diagnostics = SourceGateway(settings=settings).search_many([task])

        self.assertEqual(results[0].provider, "tavily")
        self.assertEqual(len(items), 1)
        self.assertTrue(any(item.provider == "tavily" and item.status == "used" for item in diagnostics))
        self.assertFalse(any(item.provider == "anspire" for item in diagnostics))

    def test_source_gateway_does_not_replay_unless_enabled(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY=None,
            ANSPIRE_API_KEY=None,
            BOCHA_WEB_SEARCH_API_KEY=None,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=False,
        )
        task = RetrievalTask(
            task_id="task_no_replay",
            parent_claim_id=None,
            query="DeepSeek support update",
            query_variants=[],
            target_source="web",
            purpose="source_acquisition",
            priority=1,
            deadline_sec=30,
            max_results=3,
            budget={},
            created_by="unit_test",
        )

        items, results, diagnostics = SourceGateway(settings=settings).search_many([task])

        self.assertEqual(items, [])
        self.assertEqual(results[0].provider, "tavily")
        self.assertFalse(any(item.provider == "local_fixture" for item in diagnostics))

    def test_local_replay_uses_non_http_fixture_sources(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY=None,
            ANSPIRE_API_KEY=None,
            BOCHA_WEB_SEARCH_API_KEY=None,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=True,
        )
        task = RetrievalTask(
            task_id="task_replay",
            parent_claim_id=None,
            query="DeepSeek support update",
            query_variants=[],
            target_source="web",
            purpose="source_acquisition",
            priority=1,
            deadline_sec=30,
            max_results=3,
            budget={},
            created_by="unit_test",
        )

        items, _results, diagnostics = SourceGateway(settings=settings).search_many([task])

        self.assertTrue(items)
        self.assertTrue(all(item.url.startswith("replay://") for item in items))
        self.assertEqual({item.platform for item in items}, {"local_fixture"})
        self.assertEqual({item.source_type for item in items}, {"replay_fixture"})
        self.assertTrue(any(item.provider == "local_fixture" and item.status == "used" for item in diagnostics))

    def test_external_social_web_results_use_canonical_platforms(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY="tavily-key",
            ANSPIRE_API_KEY=None,
            BOCHA_WEB_SEARCH_API_KEY=None,
            SEARCH_TIMEOUT=5,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=False,
        )
        task = RetrievalTask(
            task_id="task_social_web",
            parent_claim_id=None,
            query="DeepSeek public discussion",
            query_variants=[],
            target_source="web",
            purpose="source_acquisition",
            priority=1,
            deadline_sec=30,
            max_results=3,
            budget={},
            created_by="unit_test",
        )

        def fake_post(url, **_kwargs):
            self.assertEqual(url, "https://api.tavily.com/search")
            return FakeResponse(
                {
                    "results": [
                        {
                            "url": "https://www.reddit.com/r/LocalLLaMA/comments/example",
                            "title": "Reddit discussion",
                            "content": "Reddit users discuss support delays.",
                        },
                        {
                            "url": "https://x.com/example/status/123",
                            "title": "X post",
                            "content": "A user reports a support update on X.",
                        },
                    ]
                }
            )

        with patch("AgentCoordinator.intelligence.acquisition.source_gateway.requests.post", side_effect=fake_post):
            items, _results, _diagnostics = SourceGateway(settings=settings).search_many([task])

        by_title = {item.title: item for item in items}
        self.assertEqual(by_title["Reddit discussion"].platform, "reddit")
        self.assertEqual(by_title["X post"].platform, "twitter")
        self.assertEqual({item.source_type for item in items}, {"ugc"})
        self.assertEqual({item.acquisition_source for item in items}, {"tavily"})

    def test_web_search_executes_query_variants_and_preserves_query(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY="tavily-key",
            ANSPIRE_API_KEY=None,
            BOCHA_WEB_SEARCH_API_KEY=None,
            SEARCH_TIMEOUT=5,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=False,
        )
        task = RetrievalTask(
            task_id="task_variant_web",
            parent_claim_id=None,
            query="cost of living China",
            query_variants=["cost of living China", "rising prices China"],
            target_source="web",
            purpose="source_acquisition",
            priority=1,
            deadline_sec=30,
            max_results=3,
            budget={"max_api_calls": 2},
            created_by="unit_test",
        )
        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs["json"]["query"])
            query = kwargs["json"]["query"]
            return FakeResponse(
                {
                    "results": [
                        {
                            "url": f"https://example.com/{len(calls)}",
                            "title": f"Result for {query}",
                            "content": f"Content for {query}",
                        }
                    ]
                }
            )

        with patch("AgentCoordinator.intelligence.acquisition.source_gateway.requests.post", side_effect=fake_post):
            items, results, _diagnostics = SourceGateway(settings=settings).search_many([task])

        self.assertEqual(calls, ["cost of living China", "rising prices China"])
        self.assertEqual(results[0].items_returned, 2)
        self.assertEqual({item.retrieval_query for item in items}, set(calls))

    def test_mindspider_platform_samples_share_source_gateway_layer(self):
        settings = SimpleNamespace(
            SEARCH_TOOL_TYPE="TavilyAPI",
            TAVILY_API_KEY="tavily-key",
            ANSPIRE_API_KEY=None,
            BOCHA_WEB_SEARCH_API_KEY=None,
            SEARCH_TIMEOUT=5,
            COORDINATOR_ENABLE_MINDSPIDER_DB=True,
            COORDINATOR_ALLOW_REPLAY_FALLBACK=False,
            DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE=20,
        )
        tasks = [
            RetrievalTask(
                task_id="task_web",
                parent_claim_id=None,
                query="DeepSeek support update",
                query_variants=[],
                target_source="web",
                purpose="support",
                priority=1,
                deadline_sec=30,
                max_results=3,
                budget={},
                created_by="unit_test",
            ),
            RetrievalTask(
                task_id="task_social",
                parent_claim_id=None,
                query="DeepSeek support update",
                query_variants=[],
                target_source="mindspider_db",
                purpose="social_context",
                priority=2,
                deadline_sec=30,
                max_results=3,
                budget={},
                created_by="unit_test",
            ),
        ]

        def fake_post(url, **_kwargs):
            self.assertEqual(url, "https://api.tavily.com/search")
            return FakeResponse(
                {
                    "results": [
                        {
                            "url": "https://example.com/tavily",
                            "title": "Tavily result",
                            "content": "Tavily returned a web source about the configured topic.",
                        }
                    ]
                }
            )

        gateway = SourceGateway(settings=settings)
        with patch("AgentCoordinator.intelligence.acquisition.source_gateway.requests.post", side_effect=fake_post), patch.object(
            gateway,
            "_load_mindspider_db",
            return_value=FakeMindSpiderDB,
        ):
            items, results, diagnostics = gateway.search_many(tasks)

        self.assertEqual({result.provider for result in results}, {"tavily", "mindspider_db"})
        self.assertEqual(len(items), 2)
        self.assertTrue(any(item.platform == "weibo" and item.source_type == "ugc" for item in items))
        self.assertTrue(any(item.provider == "mindspider_db" and item.status == "used" for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
