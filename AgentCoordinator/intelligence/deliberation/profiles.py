"""Deterministic investigation briefs and versioned debate role profiles."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from AgentCoordinator.utils.perspective_templates import get_perspectives

from ..contracts import DebateAgentProfile, InvestigationBrief


COMMON_PROHIBITIONS = [
    "Do not turn sampled discourse into a population-level claim.",
    "Do not treat repeated acquisition or copied posts as independent confirmation.",
    "Do not cite model memory as factual evidence.",
]


def build_investigation_brief(query: str, target_entity: str, analysis_type: str) -> InvestigationBrief:
    topic = str(query or "").strip()
    target = str(target_entity or topic or "the target").strip()
    pricing = any(token in topic.lower() for token in ("price", "pricing", "cost", "token"))
    if pricing:
        factual = f"How did {target} pricing change according to primary or authoritative sources?"
        discourse = (
            f"What does the available sampled public discourse support about reactions to {target} pricing, "
            "without inferring population-wide opinion?"
        )
    else:
        factual = f"What material facts about {target} are directly supported by current source spans?"
        discourse = (
            f"What does the available sampled public discourse support about reactions to {target}, "
            "and where are the sample boundaries?"
        )
    obligations = {
        "empirical_fact": ["primary_source", "exact_span", "time_scope"],
        "causal_or_forecast": ["alternative_explanation", "counter_evidence", "uncertainty"],
        "discourse_observation": ["sample_boundary", "source_independence", "stance_coverage"],
        "normative_interpretation": ["stakeholder", "value_framework", "inference_label"],
    }
    return InvestigationBrief(
        original_query=topic,
        target_entity=target,
        analysis_type=analysis_type or "general",
        factual_question=factual,
        discourse_question=discourse,
        claim_modes=list(obligations),
        time_scope="retrieval_time_window",
        sample_boundary="Observed sources and platform samples are not population estimates.",
        role_obligations=obligations,
    )


def build_role_profiles(analysis_type: str, settings: Any = None) -> List[DebateAgentProfile]:
    route_map = _route_map(settings)
    profiles: List[DebateAgentProfile] = []
    for index, (name, mandate) in enumerate(get_perspectives(analysis_type), start=1):
        role_id = f"perspective_{index}_{_slug(name)}"
        obligations = _obligations(name, mandate)
        profiles.append(
            DebateAgentProfile(
                role_id=role_id,
                name=name,
                chamber="perspective",
                analytical_lens=name,
                mandate=mandate,
                evidence_obligations=obligations,
                prohibited_inferences=list(COMMON_PROHIBITIONS),
                protocol_capabilities=["position", "rebut", "revise", "request_evidence", "abstain"],
                model_route=_route_for(role_id, route_map),
                temperature=0.3,
            )
        )
    profiles.extend(
        [
            DebateAgentProfile(
                role_id="skeptic",
                name="Counter-evidence Skeptic",
                chamber="evidence_review",
                analytical_lens="adversarial counter-evidence",
                mandate="Challenge material claims with counter-evidence, alternative explanations, and missing assumptions.",
                evidence_obligations=["counter_evidence", "alternative_explanation", "source_conflict"],
                prohibited_inferences=list(COMMON_PROHIBITIONS),
                protocol_capabilities=["challenge", "qualify", "request_evidence"],
                model_route=_route_for("skeptic", route_map),
                temperature=0.2,
            ),
            DebateAgentProfile(
                role_id="methodologist",
                name="Evidence Methodologist",
                chamber="evidence_review",
                analytical_lens="sampling and inference validity",
                mandate="Check source independence, sample boundaries, temporal scope, metrics, and inference type.",
                evidence_obligations=["sample_boundary", "source_independence", "time_scope", "inference_scope"],
                prohibited_inferences=list(COMMON_PROHIBITIONS),
                protocol_capabilities=["challenge", "qualify", "request_evidence"],
                model_route=_route_for("methodologist", route_map),
                temperature=0.1,
            ),
            DebateAgentProfile(
                role_id="primary_judge",
                name="Primary Claim Judge",
                chamber="adjudication",
                analytical_lens="claim-mode rubric",
                mandate="Issue anonymous per-claim verdicts from validated evidence and argument acts.",
                evidence_obligations=["eligibility", "counter_evidence", "required_wording"],
                prohibited_inferences=list(COMMON_PROHIBITIONS),
                protocol_capabilities=["judge"],
                model_route=_route_for("primary_judge", route_map),
                temperature=0.0,
            ),
            DebateAgentProfile(
                role_id="review_judge",
                name="Review Claim Judge",
                chamber="adjudication",
                analytical_lens="order-shuffled verdict review",
                mandate="Independently re-evaluate anonymous claim subgraphs with reversed argument order.",
                evidence_obligations=["eligibility", "order_check", "unresolved_attacks"],
                prohibited_inferences=list(COMMON_PROHIBITIONS),
                protocol_capabilities=["judge"],
                model_route=_route_for("review_judge", route_map),
                temperature=0.0,
            ),
        ]
    )
    return profiles


def _obligations(name: str, mandate: str) -> List[str]:
    text = f"{name} {mandate}".lower()
    values = ["exact_span", "counter_evidence"]
    if any(token in text for token in ("public", "social", "consumer", "stakeholder")):
        values.extend(["sample_boundary", "stance_coverage"])
    if any(token in text for token in ("media", "narrative", "image")):
        values.extend(["source_framing", "source_independence"])
    if any(token in text for token in ("policy", "ethic", "safety", "institution")):
        values.extend(["value_framework", "affected_stakeholder"])
    if any(token in text for token in ("technical", "fact", "data", "economic", "feasibility")):
        values.extend(["primary_source", "time_scope"])
    return list(dict.fromkeys(values))


def _route_map(settings: Any) -> Dict[str, str]:
    raw = getattr(settings, "COORDINATOR_DEBATE_ROLE_ROUTES", "") if settings is not None else ""
    if not raw:
        return {"*": "query"}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"*": "query"}
    return {str(key): str(value) for key, value in parsed.items()} if isinstance(parsed, dict) else {"*": "query"}


def _route_for(role_id: str, route_map: Dict[str, str]) -> str:
    return route_map.get(role_id, route_map.get("perspective" if role_id.startswith("perspective_") else role_id, route_map.get("*", "query")))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:48] or "analyst"
