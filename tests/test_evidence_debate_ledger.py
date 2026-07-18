import unittest

from AgentCoordinator.intelligence.contracts import (
    AgentPosition,
    Claim,
    DebateSession,
    EvidenceGraph,
    EvidenceItem,
    InvestigationBrief,
    SourceSpan,
)
from AgentCoordinator.intelligence.deliberation import ArgumentLedger, InvalidArgumentReference


class ArgumentLedgerTestCase(unittest.TestCase):
    def setUp(self):
        brief = InvestigationBrief(
            original_query="pricing",
            target_entity="Example",
            analysis_type="technology",
            factual_question="How did pricing change?",
            discourse_question="What does sampled discourse support?",
            claim_modes=["empirical_fact", "discourse_observation"],
            time_scope="retrieval_time_window",
            sample_boundary="Sampled evidence only.",
        )
        self.session = DebateSession(
            session_id="debate_test",
            run_id="run_test",
            investigation_brief=brief,
            profiles=[],
        )
        self.graph = EvidenceGraph(
            evidence_items=[
                EvidenceItem(
                    evidence_id="ev_1",
                    item_id="item_1",
                    canonical_item_id="item_1",
                    source_type="official",
                    platform="example.com",
                    title="Official pricing",
                    text="Price changed from 0.010 to 0.015.",
                    url="https://example.com/pricing",
                    source_name="Example",
                    published_at="2026-07-01T00:00:00Z",
                    quality_ref="item_1",
                    spans=[
                        SourceSpan(
                            span_id="span_1",
                            evidence_id="ev_1",
                            text="Price changed from 0.010 to 0.015.",
                            start_char=0,
                            end_char=36,
                            span_type="source_excerpt",
                            extraction_route="test",
                            confidence=0.9,
                        )
                    ],
                )
            ],
            claims=[
                Claim(
                    claim_id="claim_1",
                    claim_text="The price changed.",
                    claim_type="fact",
                    target_entity="Example",
                    aspect="pricing",
                    time_scope="retrieval_time_window",
                    stance="official",
                    sentiment="neutral",
                    supporting_spans=["span_1"],
                    contradicting_spans=[],
                    quality_summary={"source_diversity": 0.25},
                    status="supported",
                    confidence=0.8,
                    created_by="test",
                    model="test",
                )
            ],
            blackboard_version=1,
        )

    def test_rejects_unknown_span_reference(self):
        ledger = ArgumentLedger(self.session)
        position = AgentPosition(
            position_id="position_1",
            agent_id="perspective_1",
            claim_id="claim_1",
            stance="support",
            argument="Supported by an invalid citation.",
            evidence_span_ids=["missing_span"],
            assumptions=[],
            uncertainties=[],
            confidence=0.7,
            evidence_version=1,
            round_index=0,
        )
        with self.assertRaisesRegex(InvalidArgumentReference, "Unknown evidence span"):
            ledger.add_position(position, self.graph, ["claim_1"])
        self.assertFalse(self.session.positions)


if __name__ == "__main__":
    unittest.main()
