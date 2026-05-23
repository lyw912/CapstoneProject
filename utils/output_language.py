"""
Shared output-language rules for QueryEngine and MediaEngine.

Uses project-level REPORT_OUTPUT_LANGUAGE from config (default: en).
"""

from __future__ import annotations

ENGLISH_ENGINE_OUTPUT_RULE = (
    "Language rule (mandatory when active): All user-facing prose in "
    "paragraph_latest_state, updated_paragraph_latest_state, final report bodies, "
    "and JSON string fields title/content must be written in English only. "
    "Use the English section headings exactly as specified in this prompt "
    '(e.g. "## Comprehensive Information Overview"), never Chinese translations. '
    "If source material is in Chinese, translate or paraphrase into English while "
    "preserving facts, numbers, dates, and URLs. Proper nouns (e.g., Weibo, "
    "DingXiang Doctor) may stay unchanged; surrounding narrative must be English. "
    "Do not output Chinese characters unless indispensable inside a short direct quote; "
    "prefer English paraphrase."
)


def is_english_output_mode() -> bool:
    try:
        from config import settings

        lang = str(getattr(settings, "REPORT_OUTPUT_LANGUAGE", "en") or "en").lower()
        return lang == "en"
    except Exception:
        return True


def with_output_language_rule(system_prompt: str) -> str:
    """Append English-only rule when REPORT_OUTPUT_LANGUAGE=en."""
    if not is_english_output_mode():
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n**Language rule:**\n{ENGLISH_ENGINE_OUTPUT_RULE}\n"
