"""
Report Engine LLM submodule.

Currently exposes OpenAI-compatible `LLMClient` wrapper.
"""

from .base import LLMClient

__all__ = ["LLMClient"]
