"""
Utils package for AgentCoordinator.
"""

from .timeout_guard import with_timeout
from .platform_profiles import PLATFORM_PROFILES, SOURCE_WEIGHTS
from .perspective_templates import PERSPECTIVE_TEMPLATES, get_perspectives
from .report_bridge import build_report_prompt, synthesis_context_to_markdown

__all__ = [
    "with_timeout",
    "PLATFORM_PROFILES",
    "SOURCE_WEIGHTS",
    "PERSPECTIVE_TEMPLATES",
    "get_perspectives",
    "build_report_prompt",
    "synthesis_context_to_markdown",
]
