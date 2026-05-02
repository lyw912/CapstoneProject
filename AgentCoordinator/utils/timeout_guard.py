"""
Timeout guard utility for wrapping async coroutines with a deadline.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


async def with_timeout(
    coro: Any,
    timeout_seconds: float,
    fallback: Optional[Any] = None,
    label: str = "operation",
) -> Any:
    """
    Execute an awaitable with a timeout.

    Returns `fallback` (default None) if the timeout fires or an exception occurs.
    Exceptions are not re-raised — callers should check for None to detect failures.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        from loguru import logger
        logger.warning(f"[TimeoutGuard] {label} exceeded {timeout_seconds}s timeout")
        return fallback
    except Exception as exc:
        from loguru import logger
        logger.error(f"[TimeoutGuard] {label} raised {type(exc).__name__}: {exc}")
        return fallback
