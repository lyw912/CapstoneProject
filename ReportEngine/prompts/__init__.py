"""
Report Engine Prompts Module

Centrally exports system prompts and helper functions for all workflow stages.
Other modules can import using `from prompts import ...`.
"""

from .prompts import (
    SYSTEM_PROMPT_TEMPLATE_SELECTION,
    SYSTEM_PROMPT_HTML_GENERATION,
    SYSTEM_PROMPT_CHAPTER_JSON,
    SYSTEM_PROMPT_CHAPTER_JSON_REPAIR,
    SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY,
    SYSTEM_PROMPT_DOCUMENT_LAYOUT,
    SYSTEM_PROMPT_WORD_BUDGET,
    output_schema_template_selection,
    input_schema_html_generation,
    chapter_generation_input_schema,
    build_chapter_user_prompt,
    build_chapter_repair_prompt,
    build_chapter_recovery_payload,
    build_document_layout_prompt,
    build_word_budget_prompt,
)

__all__ = [
    "SYSTEM_PROMPT_TEMPLATE_SELECTION",
    "SYSTEM_PROMPT_HTML_GENERATION",
    "SYSTEM_PROMPT_CHAPTER_JSON",
    "SYSTEM_PROMPT_CHAPTER_JSON_REPAIR",
    "SYSTEM_PROMPT_DOCUMENT_LAYOUT",
    "SYSTEM_PROMPT_WORD_BUDGET",
    "SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY",
    "output_schema_template_selection",
    "input_schema_html_generation",
    "chapter_generation_input_schema",
    "build_chapter_user_prompt",
    "build_chapter_repair_prompt",
    "build_chapter_recovery_payload",
    "build_document_layout_prompt",
    "build_word_budget_prompt",
]
