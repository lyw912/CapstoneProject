import importlib.util
import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


academic_module = _load_module("academic_report_generator", "AgentCoordinator/academic_report_generator.py")
report_bridge_module = _load_module("report_bridge", "AgentCoordinator/utils/report_bridge.py")

generate_academic_report = academic_module.generate_academic_report
ENGLISH_OUTPUT_CONSTRAINT = report_bridge_module.ENGLISH_OUTPUT_CONSTRAINT
coordinator_output_to_report_engine_inputs = report_bridge_module.coordinator_output_to_report_engine_inputs
generate_report_engine_html = report_bridge_module.generate_report_engine_html


class FakeReportAgent:
    def __init__(self):
        self.kwargs = None

    def generate_report(self, **kwargs):
        self.kwargs = kwargs
        return {"html_content": "<html>ok</html>"}


class CoordinatorReportBridgeTestCase(unittest.TestCase):
    def _sample_output(self):
        return {
            "schema_version": "1.0",
            "query": "DeepSeek new model public opinion",
            "analysis_type": "technology",
            "generated_at": "2026-05-03T00:00:00+00:00",
            "pipeline_duration_seconds": 12.5,
            "divergence_matrix": {
                "pairs": {"web|weibo": 0.42},
                "hotspots": ["web and weibo diverge on confidence"],
                "max_divergence": {"pair": "web|weibo", "value": 0.42},
                "min_divergence": {"pair": "web|weibo", "value": 0.42},
            },
            "deliberation": {
                "analysis_type": "technology",
                "perspectives_used": ["Technical facts", "Public sentiment"],
                "phases": [],
                "final_consensus": ["Evidence supports strong attention but mixed trust."],
                "final_dissents": ["Performance claims remain disputed."],
                "confidence": 0.71,
            },
            "gap_filling": {"rounds_performed": 1, "gaps_detected": [], "results_found": 2},
            "platform_interpretations": {"weibo": "Weibo discussion is more emotional."},
            "bias_analysis": {
                "echo_warnings": ["Social samples are platform-biased."],
                "silent_majority_hypothesis": None,
            },
            "fact_opinion_separation": {
                "verified_facts": [
                    {
                        "fact": "The model announcement generated cross-platform discussion.",
                        "sources": ["https://example.com/news"],
                        "verification_status": "single_source",
                        "confidence": 0.8,
                    }
                ],
                "opinions_sentiments": [],
                "analytical_frameworks": [],
            },
            "synthesis": {
                "summary": "The discourse is attentive but not uniformly supportive.",
                "top_insights": [],
                "key_tensions": [],
                "overall_confidence": 0.72,
                "recommended_investigation": ["Collect more benchmark evidence."],
            },
            "source_data": {
                "query_agent": {
                    "total_sources": 3,
                    "stance_distribution": {"support": 0.4, "neutral": 0.6},
                    "coverage_score": 0.8,
                    "top_sources": [
                        {
                            "title": "Announcement analysis",
                            "url": "https://example.com/news",
                            "trust_score": 0.91,
                            "stance": "neutral",
                        }
                    ],
                    "social_sentiment": {
                        "mode": "available",
                        "platforms_queried": ["weibo"],
                        "total_posts": 2,
                        "total_comments": 1,
                        "sentiment_distribution": {"support": 0.5, "neutral": 0.5},
                        "divergence_score": 0.42,
                        "top_social_voices": [
                            {
                                "platform": "weibo",
                                "stance": "support",
                                "content": "这是一个中文原文证据。",
                                "url": "https://weibo.example/post/1",
                                "publish_time": "2026-05-03T00:00:00",
                            }
                        ],
                    },
                },
                "media_agent": {"available": True, "mode": "live", "summary_length": 100},
            },
            "coordinator_trace": ["trace entry"],
            "agent_errors": [],
        }

    def test_adapter_builds_report_engine_inputs(self):
        output = self._sample_output()
        inputs = coordinator_output_to_report_engine_inputs(output)

        self.assertEqual(inputs["query"], output["query"])
        self.assertEqual(len(inputs["reports"]), 2)
        self.assertIn("Binding Evidence Policy", inputs["reports"][0])
        self.assertNotIn("Structured Coordinator Output", inputs["reports"][0])
        self.assertLessEqual(len(inputs["reports"][0]), 30000)
        self.assertIn("Representative Social Voices", inputs["reports"][1])
        self.assertIn(ENGLISH_OUTPUT_CONSTRAINT, inputs["forum_logs"])
        self.assertIn("Multi-Source Public Opinion Analysis Report", inputs["custom_template"])

    def test_evidence_bound_package_uses_strictest_paired_wording(self):
        output = self._sample_output()
        output["investigation_brief"] = {
            "original_query": output["query"],
            "factual_question": "What changed?",
            "discourse_question": "What does the observed sample support?",
            "time_scope": "retrieval window",
            "sample_boundary": "Observed sources are not population estimates.",
        }
        output["coordinator_intelligence"] = {
            "provider_diagnostics": [
                {
                    "provider": "media_agent",
                    "capability": "specialist_llm",
                    "status": "disabled",
                    "model": "deepseek-chat",
                }
            ],
            "evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim_material",
                        "claim_text": "Pricing is always lower than every competitor.",
                        "supporting_spans": ["span_1"],
                        "contradicting_spans": [],
                    },
                    {
                        "claim_id": "claim_rejected",
                        "claim_text": "A rejected assertion.",
                        "supporting_spans": ["span_1"],
                        "contradicting_spans": [],
                    },
                ],
                "audit_decisions": [],
                "evidence_items": [
                    {
                        "title": "Authoritative pricing source",
                        "url": "https://example.com/pricing",
                        "platform": "example.com",
                        "source_type": "official",
                        "spans": [{"span_id": "span_1", "text": "A bounded price observation."}],
                    }
                ],
            },
        }
        output["debate"] = {
            "material_claims": [
                {"claim_id": "claim_material", "score": 6.5, "reason_codes": ["high_inference_claim"]},
                {"claim_id": "claim_rejected", "score": 5.5, "reason_codes": ["missing_evidence"]},
            ],
            "positions": [
                {"claim_id": "claim_material", "agent_id": "technical", "stance": "support"}
            ],
            "argument_acts": [
                {
                    "target_claim_id": "claim_material",
                    "actor_id": "skeptic",
                    "reason_codes": ["missing_evidence"],
                }
            ],
            "revisions": [
                {"claim_id": "claim_material", "revision_type": "revise"}
            ],
            "verdicts": [
                {
                    "claim_id": "claim_material",
                    "judge_id": "review_judge",
                    "decision": "accept",
                    "final_wording": "Observed sources report lower pricing at specific points.",
                    "required_edit": "accept_revision",
                    "confidence": 0.8,
                    "evidence_span_ids": ["span_1"],
                },
                {
                    "claim_id": "claim_material",
                    "judge_id": "primary_judge",
                    "decision": "weaken",
                    "final_wording": "Observed sources report lower pricing at specific points.",
                    "required_edit": "Remove the universal comparison.",
                    "confidence": 0.9,
                    "evidence_span_ids": ["span_1"],
                },
                {
                    "claim_id": "claim_rejected",
                    "judge_id": "primary_judge",
                    "decision": "reject",
                    "final_wording": None,
                    "required_edit": "remove",
                    "confidence": 0.95,
                    "evidence_span_ids": ["span_1"],
                },
            ],
            "output_groups": {
                "audited_findings": ["claim_material"],
                "contested_findings": [],
                "perspective_tensions": [],
                "rejected_claims": ["claim_rejected"],
                "evidence_gaps": [],
            },
        }

        inputs = coordinator_output_to_report_engine_inputs(output)
        query_report = inputs["reports"][0]

        self.assertIn("Observed sources report lower pricing at specific points.", query_report)
        self.assertNotIn("Pricing is always lower than every competitor.", query_report)
        self.assertIn("primary_judge=weaken", query_report)
        self.assertIn("DO NOT REPORT AS A FACT", query_report)
        self.assertIn("[Authoritative pricing source](https://example.com/pricing)", query_report)
        self.assertIn("media_agent: status=disabled", query_report)
        self.assertLessEqual(len(query_report), 30000)

    def test_report_engine_runner_forwards_adapter_inputs(self):
        fake = FakeReportAgent()
        result = generate_report_engine_html(self._sample_output(), report_agent=fake, save_report=False)

        self.assertEqual(result["html_content"], "<html>ok</html>")
        self.assertFalse(fake.kwargs["save_report"])
        self.assertEqual(fake.kwargs["query"], "DeepSeek new model public opinion")
        self.assertIn(ENGLISH_OUTPUT_CONSTRAINT, fake.kwargs["forum_logs"])
        self.assertIn("Multi-Source Public Opinion Analysis Report", fake.kwargs["custom_template"])
        self.assertEqual(result["adapter_metadata"]["language_constraint"], ENGLISH_OUTPUT_CONSTRAINT)

    def test_binding_policy_blocks_report_without_final_wording_and_citation(self):
        output = self._sample_output()
        output["coordinator_intelligence"] = {
            "provider_diagnostics": [],
            "evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim_material",
                        "claim_text": "Observed sources report a bounded price change.",
                        "supporting_spans": ["span_1"],
                        "contradicting_spans": [],
                    }
                ],
                "audit_decisions": [],
                "evidence_items": [
                    {
                        "title": "Bound source",
                        "url": "https://example.com/bound",
                        "spans": [{"span_id": "span_1", "text": "Bound excerpt"}],
                    }
                ],
            },
        }
        output["debate"] = {
            "material_claims": [{"claim_id": "claim_material"}],
            "verdicts": [
                {
                    "claim_id": "claim_material",
                    "judge_id": "primary_judge",
                    "decision": "weaken",
                    "final_wording": "Observed sources report a bounded price change.",
                    "evidence_span_ids": ["span_1"],
                }
            ],
            "output_groups": {"audited_findings": ["claim_material"]},
        }
        fake = FakeReportAgent()

        with self.assertRaises(report_bridge_module.EvidencePolicyViolation):
            generate_report_engine_html(output, report_agent=fake, save_report=True)

        self.assertFalse(fake.kwargs["save_report"])

    def test_binding_policy_accepts_exact_wording_with_bound_link(self):
        output = self._sample_output()
        output["coordinator_intelligence"] = {
            "provider_diagnostics": [],
            "evidence_graph": {
                "claims": [
                    {
                        "claim_id": "claim_material",
                        "claim_text": "Observed sources report a bounded price change.",
                        "supporting_spans": ["span_1"],
                        "contradicting_spans": [],
                    }
                ],
                "audit_decisions": [],
                "evidence_items": [
                    {
                        "title": "Bound source",
                        "url": "https://example.com/bound",
                        "spans": [{"span_id": "span_1", "text": "Bound excerpt"}],
                    }
                ],
            },
        }
        output["debate"] = {
            "material_claims": [{"claim_id": "claim_material"}],
            "verdicts": [
                {
                    "claim_id": "claim_material",
                    "judge_id": "primary_judge",
                    "decision": "weaken",
                    "final_wording": "Observed sources report a bounded price change.",
                    "evidence_span_ids": ["span_1"],
                }
            ],
            "output_groups": {"audited_findings": ["claim_material"]},
        }
        html = (
            "<html><body><p>Observed sources report a bounded price change.</p>"
            "<a href=\"https://example.com/bound\">Bound source</a></body></html>"
        )

        compliance = report_bridge_module._validate_generated_report(output, html)

        self.assertTrue(compliance["passed"])
        self.assertEqual(compliance["checked_reportable_claims"], 1)

    def test_report_agent_node_uses_canonical_schema_builder(self):
        source = (PROJECT_ROOT / "AgentCoordinator/graph/nodes/report_agent_node.py").read_text(encoding="utf-8")

        self.assertIn("build_coordinator_output", source)
        self.assertIn("duration_seconds=0.0", source)
        self.assertNotIn("top_sources = sorted", source)
        self.assertNotIn('"source_data": {', source)

    def test_academic_report_preserves_only_original_chinese_evidence(self):
        report = generate_academic_report(self._sample_output())

        self.assertIn("## Abstract", report)
        self.assertIn("Representative Social Media Voices", report)
        self.assertIn("这是一个中文原文证据", report)
        self.assertNotIn("网络媒体层", report)
        self.assertNotIn("分析管线版本", report)

    def test_report_engine_language_directives_are_english(self):
        prompt_source = (PROJECT_ROOT / "ReportEngine/prompts/prompts.py").read_text(encoding="utf-8")
        agent_source = (PROJECT_ROOT / "ReportEngine/agent.py").read_text(encoding="utf-8")

        self.assertIn("must be written in English", prompt_source)
        self.assertIn("Arabic numbering", prompt_source)
        self.assertNotIn('language": "zh-CN"', prompt_source)
        self.assertIn('"language": "en-US"', agent_source)
        self.assertIn("ENGLISH_REPORT_LANGUAGE_RULE", agent_source)


if __name__ == "__main__":
    unittest.main()
