"""
Executable JSON Contract (IR) definition and validation tools for Report Engine.

This module exposes unified Schema text and validators for prompts, chapter generation,
and final stitching workflows to ensure consistent output structure from LLM to rendering.
"""

from .schema import (
    IR_VERSION,
    CHAPTER_JSON_SCHEMA,
    CHAPTER_JSON_SCHEMA_TEXT,
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    ENGINE_AGENT_TITLES,
)
from .validator import IRValidator

__all__ = [
    "IR_VERSION",
    "CHAPTER_JSON_SCHEMA",
    "CHAPTER_JSON_SCHEMA_TEXT",
    "ALLOWED_BLOCK_TYPES",
    "ALLOWED_INLINE_MARKS",
    "ENGINE_AGENT_TITLES",
    "IRValidator",
]
