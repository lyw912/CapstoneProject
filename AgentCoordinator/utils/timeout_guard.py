"""
Timeout guard utility for wrapping async coroutines with a deadline.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    fallback: Optional[T] = None,
    label: str = "operation",
) -> Optional[T]:
    """
    Execute an awaitable with a timeout.

    Returns `fallback` (default None) if the timeout fires or an exception occurs.
    Callers running blocking agent graphs should wrap sync work in asyncio.to_thread
    so the event loop can honor deadlines while Query/Media run in parallel.
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


async def run_sync_with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args: Any,
    fallback: Optional[T] = None,
    label: str = "operation",
    **kwargs: Any,
) -> Optional[T]:
    """Run blocking sync code in a worker thread with an asyncio deadline."""
    return await with_timeout(
        asyncio.to_thread(func, *args, **kwargs),
        timeout_seconds=timeout_seconds,
        fallback=fallback,
        label=label,
    )
