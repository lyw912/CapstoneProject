"""
Shared helpers for per-engine LLM HTTP timeouts (seconds).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def _coerce_positive(value: object, default: float) -> float:
    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def resolve_llm_timeouts(
    settings: Optional[object] = None,
    engine_env_key: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Return (short_task_timeout, long_task_timeout) in seconds.

    Precedence for long timeout:
      settings.LLM_LONG_TASK_TIMEOUT -> env engine key -> LLM_REQUEST_TIMEOUT -> 600
  Precedence for short timeout:
      settings.LLM_SHORT_TASK_TIMEOUT -> 120
    """
    if settings is None:
        from config import settings as settings  # noqa: WPS440

    short_default = _coerce_positive(
        getattr(settings, "LLM_SHORT_TASK_TIMEOUT", 120),
        120.0,
    )

    long_from_settings = getattr(settings, "LLM_LONG_TASK_TIMEOUT", None)
    long_from_env = None
    if engine_env_key:
        long_from_env = os.getenv(engine_env_key)
    long_fallback = os.getenv("LLM_REQUEST_TIMEOUT")
    long_default = _coerce_positive(
        long_from_settings or long_from_env or long_fallback,
        600.0,
    )

    return short_default, long_default


def resolve_stream_idle_timeout(settings: Optional[object] = None) -> float:
    """Return idle timeout (seconds) between streaming chunks."""
    if settings is None:
        from config import settings as settings  # noqa: WPS440

    return _coerce_positive(
        getattr(settings, "LLM_STREAM_IDLE_TIMEOUT", 240),
        240.0,
    )
