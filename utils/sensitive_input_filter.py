"""
User-input sensitive word filter.

Checks free-text fields supplied by operators (query, template, feedback, etc.)
before starting analysis or report generation. Does not scan engine-generated content.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from loguru import logger

SENSITIVE_INPUT_MESSAGE = (
    "Your input contains blocked terms, so the report cannot be generated. "
    "Please revise the topic and try again."
)
SENSITIVE_INPUT_ERROR_CODE = "sensitive_input"

_FULLWIDTH_ASCII_OFFSET = 0xFEE0


@dataclass(frozen=True)
class SensitiveInputResult:
    blocked: bool
    field: Optional[str] = None


def _normalize_text(text: str) -> str:
    """Normalize user text for consistent matching."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    chars = []
    for ch in normalized:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - _FULLWIDTH_ASCII_OFFSET))
        else:
            chars.append(ch)
    collapsed = re.sub(r"\s+", "", "".join(chars))
    return collapsed.casefold()


def _load_words_from_file(path: Path) -> Tuple[str, ...]:
    if not path.exists():
        logger.warning("Sensitive words file not found: {}", path)
        return tuple()
    words = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    return tuple(words)


@lru_cache(maxsize=4)
def _cached_words(file_path: str, mtime_ns: int) -> Tuple[str, ...]:
    del mtime_ns
    return _load_words_from_file(Path(file_path))


def get_sensitive_words(words_file: Path) -> Tuple[str, ...]:
    resolved = words_file.resolve()
    if not resolved.exists():
        return tuple()
    stat = resolved.stat()
    return _cached_words(str(resolved), stat.st_mtime_ns)


def contains_sensitive_word(text: str, words: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    for word in words:
        candidate = _normalize_text(word)
        if candidate and candidate in normalized:
            return True
    return False


def check_sensitive_input(
    text: str,
    *,
    enabled: bool = True,
    words_file: Optional[Path] = None,
) -> bool:
    """Return True when the text should be blocked."""
    if not enabled:
        return False
    if not str(text or "").strip():
        return False
    path = words_file or Path("config/sensitive_words.txt")
    words = get_sensitive_words(path)
    if not words:
        return False
    return contains_sensitive_word(text, words)


def check_sensitive_fields(
    fields: Dict[str, str],
    *,
    enabled: bool = True,
    words_file: Optional[Path] = None,
) -> SensitiveInputResult:
    """Check multiple named user-input fields; returns the first blocked field."""
    if not enabled:
        return SensitiveInputResult(blocked=False)
    path = words_file or Path("config/sensitive_words.txt")
    words = get_sensitive_words(path)
    if not words:
        return SensitiveInputResult(blocked=False)
    for name, value in fields.items():
        if value and contains_sensitive_word(value, words):
            return SensitiveInputResult(blocked=True, field=name)
    return SensitiveInputResult(blocked=False)


def sensitive_input_payload(field: str = "query") -> dict:
    return {
        "success": False,
        "error_code": SENSITIVE_INPUT_ERROR_CODE,
        "message": SENSITIVE_INPUT_MESSAGE,
        "field": field,
    }


def filter_settings_from_config(config) -> Tuple[bool, Path]:
    enabled = bool(getattr(config, "ENABLE_SENSITIVE_INPUT_FILTER", True))
    words_file = Path(getattr(config, "SENSITIVE_WORDS_FILE", "config/sensitive_words.txt"))
    return enabled, words_file


def reject_if_sensitive(fields: Dict[str, str], config) -> Optional[dict]:
    """Return an API error payload when any field matches the sensitive word list."""
    enabled, words_file = filter_settings_from_config(config)
    result = check_sensitive_fields(fields, enabled=enabled, words_file=words_file)
    if result.blocked:
        return sensitive_input_payload(result.field or "query")
    return None
