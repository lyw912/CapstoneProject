"""
Idle and total wall-clock watchdogs for blocking LLM stream iterators.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Generator, Iterable, TypeVar

T = TypeVar("T")


class StreamIdleTimeoutError(TimeoutError):
    """No stream chunk received within the idle window."""


class StreamTotalTimeoutError(TimeoutError):
    """Stream exceeded the overall wall-clock budget."""


def iter_with_idle_timeout(
    stream: Iterable[T],
    *,
    idle_timeout: float,
    total_timeout: float | None = None,
    poll_interval: float = 0.5,
) -> Generator[T, None, None]:
    """
    Yield items from a blocking stream iterator, aborting when:
      - no item arrives for ``idle_timeout`` seconds (time-to-first-token included), or
      - ``total_timeout`` seconds elapse since the stream started (optional).
    """
    if idle_timeout <= 0:
        yield from stream
        return

    item_queue: queue.Queue = queue.Queue()

    def _reader() -> None:
        try:
            for item in stream:
                item_queue.put(("item", item))
            item_queue.put(("done", None))
        except BaseException as exc:  # noqa: BLE001 — propagate any stream failure
            item_queue.put(("error", exc))

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    started = time.monotonic()
    last_activity = started

    while True:
        now = time.monotonic()
        if total_timeout is not None and now - started >= total_timeout:
            raise StreamTotalTimeoutError(
                f"Stream exceeded total timeout of {total_timeout:.0f}s"
            )
        if now - last_activity >= idle_timeout:
            raise StreamIdleTimeoutError(
                f"No stream data for {idle_timeout:.0f}s"
            )

        wait_budget = idle_timeout - (now - last_activity)
        if total_timeout is not None:
            wait_budget = min(wait_budget, total_timeout - (now - started))
        wait_budget = max(0.1, min(wait_budget, poll_interval))

        try:
            kind, payload = item_queue.get(timeout=wait_budget)
        except queue.Empty:
            continue

        if kind == "item":
            last_activity = time.monotonic()
            yield payload
        elif kind == "done":
            break
        elif kind == "error":
            raise payload
