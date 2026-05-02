"""
Prompts package for AgentCoordinator.
"""

from .deliberation_prompts import (
    INDEPENDENT_ANALYSIS_PROMPT,
    CROSS_EXAMINATION_PROMPT,
    SYNTHESIS_ARBITRATION_PROMPT,
)
from .fact_separation_prompt import FACT_OPINION_SEPARATION_PROMPT
from .synthesis_prompt import SYNTHESIS_PROMPT

__all__ = [
    "INDEPENDENT_ANALYSIS_PROMPT",
    "CROSS_EXAMINATION_PROMPT",
    "SYNTHESIS_ARBITRATION_PROMPT",
    "FACT_OPINION_SEPARATION_PROMPT",
    "SYNTHESIS_PROMPT",
]
