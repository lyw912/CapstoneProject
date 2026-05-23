"""
Utils package for AgentCoordinator.
"""

from .timeout_guard import with_timeout
from .platform_profiles import PLATFORM_PROFILES, SOURCE_WEIGHTS
from .perspective_templates import PERSPECTIVE_TEMPLATES, get_perspectives
from .report_bridge import (
    ENGLISH_OUTPUT_CONSTRAINT,
    build_english_report_template,
    build_report_prompt,
    coordinator_output_to_report_engine_inputs,
    generate_report_engine_html,
    synthesis_context_to_markdown,
)

__all__ = [
    "with_timeout",
    "PLATFORM_PROFILES",
    "SOURCE_WEIGHTS",
    "PERSPECTIVE_TEMPLATES",
    "get_perspectives",
    "ENGLISH_OUTPUT_CONSTRAINT",
    "build_english_report_template",
    "build_report_prompt",
    "coordinator_output_to_report_engine_inputs",
    "generate_report_engine_html",
    "synthesis_context_to_markdown",
]
