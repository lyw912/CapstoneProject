"""
Translate ReportEngine inputs from Chinese to English before generation.

Used when REPORT_OUTPUT_LANGUAGE=en so upstream Chinese Query/Media/forum text
does not steer chapter LLMs back into Chinese prose.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from ReportEngine.llms import LLMClient

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_CHUNK_SIZE = 8000

_TRANSLATE_SYSTEM_PROMPT = """You are a professional translator for public-opinion and research reports.
Translate the user's text into clear English.

Rules:
- Preserve all facts, numbers, dates, statistics, and URLs exactly.
- Keep platform and product proper nouns (e.g., Weibo, Bilibili, DeepSeek) unchanged.
- Do not add commentary, headings, or markdown fences unless they were in the source.
- Output only the English translation."""


def contains_cjk(text: str) -> bool:
    """Return True if the string contains CJK unified ideographs."""
    return bool(text and _CJK_PATTERN.search(text))


def translate_to_english(
    llm_client: LLMClient,
    text: str,
    *,
    label: str = "text",
    enabled: bool = True,
) -> str:
    """
    Translate text to English when enabled and CJK characters are present.

    On failure or empty LLM response, returns the original text.
    """
    if not enabled or not text or not contains_cjk(text):
        return text

    chunks = _split_chunks(text)
    translated_parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        user_prompt = chunk
        if len(chunks) > 1:
            user_prompt = (
                f"Translate part {index}/{len(chunks)} of a longer document ({label}).\n\n"
                f"{chunk}"
            )
        try:
            result = llm_client.invoke(
                _TRANSLATE_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.1,
                top_p=0.9,
            )
        except Exception as exc:
            logger.warning(
                "Input translation failed for {label} (chunk {index}/{total}): {error}",
                label=label,
                index=index,
                total=len(chunks),
                error=exc,
            )
            return text

        cleaned = (result or "").strip()
        if not cleaned:
            logger.warning(
                "Input translation returned empty for {label} (chunk {index}/{total}), keeping original",
                label=label,
                index=index,
                total=len(chunks),
            )
            return text
        translated_parts.append(cleaned)

    return "\n\n".join(translated_parts)


def _split_chunks(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """Split long text on paragraph boundaries to stay within LLM context limits."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        piece = paragraph if paragraph else ""
        addition = len(piece) + (2 if current else 0)
        if current and current_len + addition > chunk_size:
            chunks.append("\n\n".join(current))
            current = [piece] if piece else []
            current_len = len(piece)
            continue
        if piece:
            current.append(piece)
            current_len += addition
        elif not current:
            continue

    if current:
        chunks.append("\n\n".join(current))

    if not chunks:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
