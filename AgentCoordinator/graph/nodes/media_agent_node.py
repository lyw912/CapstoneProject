"""MediaAgentNode: best-effort live MediaEngine invocation.

MediaAgent is optional for the Coordinator pipeline. If it is configured, this
node calls the real MediaEngine agent. If its API/search keys are missing,
imports fail, or execution times out, the node records a trace-only skip and
lets the rest of the Coordinator continue with the available agents.

Caching: After the first successful run, Markdown output is saved to
AgentCoordinator/cache/. Subsequent runs load from cache (keyed by query hash).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ..state import CoordinatorState, AgentRunResult
from ...utils.timeout_guard import run_sync_with_timeout

CACHE_DIR = Path(__file__).resolve().parents[3] / "AgentCoordinator" / "cache"


def _media_agent_timeout_seconds() -> float:
    return max(60.0, float(getattr(settings, "COORDINATOR_MEDIA_AGENT_TIMEOUT", 3600) or 3600))


def _cache_path(query: str) -> Path:
    key = hashlib.md5(query.encode()).hexdigest()[:12]
    return CACHE_DIR / f"media_agent_{key}.md"


def _load_cache(query: str) -> Optional[str]:
    path = _cache_path(query)
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                logger.info(f"[MediaAgentNode] Cache hit: {path.name} ({len(text)} chars)")
                return text
        except Exception as exc:
            logger.warning(f"[MediaAgentNode] Cache load failed: {exc}")
    return None


def _save_cache(query: str, markdown: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(query)
    try:
        path.write_text(markdown, encoding="utf-8")
        logger.info(f"[MediaAgentNode] Cached result → {path.name}")
    except Exception as exc:
        logger.warning(f"[MediaAgentNode] Cache save failed: {exc}")


def _missing_media_config() -> list[str]:
    """Return missing config keys required for a live MediaAgent call."""
    missing: list[str] = []
    if not settings.MEDIA_ENGINE_API_KEY:
        missing.append("MEDIA_ENGINE_API_KEY")

    search_tool = (settings.SEARCH_TOOL_TYPE or "AnspireAPI").lower()
    if search_tool == "bochaapi":
        if not (getattr(settings, "BOCHA_API_KEY", None) or settings.BOCHA_WEB_SEARCH_API_KEY):
            missing.append("BOCHA_WEB_SEARCH_API_KEY")
    else:
        if not settings.ANSPIRE_API_KEY:
            missing.append("ANSPIRE_API_KEY")

    return missing


def _run_media_agent_sync(query: str) -> Optional[str]:
    """Invoke MediaAgent on a worker thread. Returns Markdown string or None."""
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        if (settings.SEARCH_TOOL_TYPE or "").lower() == "anspireapi":
            from MediaEngine.agent import AnspireSearchAgent as MediaAgent
        else:
            from MediaEngine.agent import DeepSearchAgent as MediaAgent

        agent = MediaAgent()
        return agent.research(query)
    except Exception as exc:
        logger.info(f"[MediaAgentNode] MediaAgent skipped: {exc}")
        return None


async def media_agent_node(state: CoordinatorState) -> dict:
    """LangGraph node: execute MediaAgent when configured, otherwise skip quietly."""
    query = state["query"]
    logger.info(f"[MediaAgentNode] Starting for query: {query!r}")
    t0 = time.time()

    cached = _load_cache(query)
    if cached is not None:
        duration = time.time() - t0
        run_result: AgentRunResult = {
            "agent_name": "media_agent",
            "success": True,
            "output": None,
            "text_output": cached,
            "error": None,
            "duration_seconds": duration,
        }
        trace = f"[MediaAgentNode] Loaded from cache in {duration:.1f}s ({len(cached)} chars)"
        logger.info(trace)
        return {"media_run": run_result, "coordinator_trace": [trace]}

    text_output: Optional[str] = None
    missing_config = _missing_media_config()

    if missing_config:
        mode = "skipped_unconfigured"
        detail = f"missing config: {', '.join(missing_config)}"
        logger.info(f"[MediaAgentNode] {mode} — {detail}")
    else:
        timeout_seconds = _media_agent_timeout_seconds()
        logger.info(f"[MediaAgentNode] Timeout budget: {timeout_seconds:.0f}s")
        text_output = await run_sync_with_timeout(
            _run_media_agent_sync,
            timeout_seconds,
            query,
            label="MediaAgent.research",
        )
        if text_output:
            _save_cache(query, text_output)
            mode = "live"
            detail = f"{len(text_output)} chars"
        else:
            mode = "skipped_unavailable"
            detail = "MediaAgent returned no output"

    duration = time.time() - t0
    agent_success = bool(text_output) if not missing_config else False

    run_result: AgentRunResult = {
        "agent_name": "media_agent",
        "success": agent_success,
        "output": None,
        "text_output": text_output,
        "error": None if agent_success else detail,
        "duration_seconds": duration,
    }

    trace = f"[MediaAgentNode] {mode} — {detail} in {duration:.1f}s"
    logger.info(trace)

    return {
        "media_run": run_result,
        "coordinator_trace": [trace],
    }
