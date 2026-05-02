"""
Dynamic perspective templates based on analysis_type.

Maps each topic type to 4 deliberation dimensions with English names.
"""

from typing import Dict, List, Tuple

# (dimension_name, role_description)
PERSPECTIVE_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "event": [
        (
            "Factual Verification",
            "You focus exclusively on verifiable facts: dates, figures, official statements, "
            "technical parameters. Anchor all claims to specific source URLs.",
        ),
        (
            "Social Impact & Emotional Response",
            "You analyze how the event affected people emotionally and socially: "
            "what groups are impacted, how they feel, and why their reactions matter.",
        ),
        (
            "Policy & Institutional Reflection",
            "You examine policy, legal, and systemic dimensions: "
            "regulatory responses, institutional failures or successes, and governance implications.",
        ),
        (
            "Historical Analogy & Trend Analysis",
            "You draw on historical precedents and long-term trend data to contextualize the event, "
            "identifying patterns and predicting likely trajectories.",
        ),
    ],
    "brand": [
        (
            "Consumer Experience",
            "You represent actual consumers: product quality, service experience, price-value ratio, "
            "emotional connection to the brand. Ground analysis in user-reported data.",
        ),
        (
            "Business Strategy & Competition",
            "You analyze from an industry and competitive strategy perspective: "
            "market positioning, competitor moves, financial implications, and strategic decisions.",
        ),
        (
            "Media Narrative & Framing",
            "You examine how media and influencers are framing the brand story: "
            "PR angles, narrative choices, what is emphasized vs omitted, and reputational impact.",
        ),
        (
            "Investor & Market Reaction",
            "You focus on financial markets and investor sentiment: "
            "stock movements, analyst assessments, earnings implications, and capital allocation signals.",
        ),
    ],
    "policy": [
        (
            "Policymaker Perspective",
            "You represent the policy design rationale: "
            "stated objectives, implementation mechanisms, trade-offs considered, and success metrics.",
        ),
        (
            "Affected Stakeholder Perspective",
            "You voice the groups directly impacted by the policy: "
            "how their lives change, costs and benefits they experience, and unintended consequences.",
        ),
        (
            "Economic & Technical Feasibility",
            "You assess whether the policy can work: "
            "cost estimates, implementation capacity, technical requirements, and precedent outcomes.",
        ),
        (
            "International Comparison & Historical Evidence",
            "You compare with international examples and historical evidence: "
            "what worked elsewhere, failure modes to avoid, and lessons from past policies.",
        ),
    ],
    "technology": [
        (
            "Technical Facts & Engineering Reality",
            "You anchor the discussion in technical truth: "
            "how the technology actually works, current capability benchmarks, and engineering constraints.",
        ),
        (
            "Industry Application & Commercial Impact",
            "You analyze business and industry implications: "
            "adoption curves, market disruption, competitive dynamics, and ROI considerations.",
        ),
        (
            "Public Perception & Emotional Reaction",
            "You examine how the general public understands and feels about the technology: "
            "fears, excitement, misconceptions, and cultural attitudes.",
        ),
        (
            "Ethics, Safety & Societal Impact",
            "You examine ethical dimensions, safety risks, and broader societal consequences: "
            "fairness, privacy, labor displacement, environmental impact, and governance needs.",
        ),
    ],
    "general": [
        (
            "Facts & Data",
            "You focus on verifiable facts, statistics, and primary source evidence. "
            "All claims must be anchored to specific sources from the provided data.",
        ),
        (
            "Public Emotion & Social Psychology",
            "You analyze collective emotional states, psychological drivers, "
            "and how social dynamics shape the public discourse.",
        ),
        (
            "Stakeholder Analysis",
            "You map who has a stake in this topic, what their interests are, "
            "how their positions differ, and what power dynamics are at play.",
        ),
        (
            "Historical & Cross-Domain Reflection",
            "You bring historical perspective and cross-domain frameworks: "
            "analogies, philosophical context, and interdisciplinary insights.",
        ),
    ],
    "person": [
        (
            "Factual Record & Achievements",
            "You focus on documented facts about the person: "
            "verified career history, concrete achievements, and public record.",
        ),
        (
            "Public Image & Reputation",
            "You analyze how the person is perceived across different audiences: "
            "supporter narratives, critics' narratives, and media framing patterns.",
        ),
        (
            "Stakeholder Impact",
            "You examine how this person's actions affect different stakeholder groups: "
            "employees, communities, industries, or political constituents.",
        ),
        (
            "Contextual & Comparative Analysis",
            "You situate the person within broader historical, cultural, or industry context: "
            "comparable figures, structural factors, and systemic forces.",
        ),
    ],
}


def get_perspectives(analysis_type: str) -> List[Tuple[str, str]]:
    """Return the 4 perspective (name, role_description) pairs for a given analysis_type."""
    return PERSPECTIVE_TEMPLATES.get(analysis_type, PERSPECTIVE_TEMPLATES["general"])
