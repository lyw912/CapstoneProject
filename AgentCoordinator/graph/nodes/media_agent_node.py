"""MediaAgentNode: best-effort live MediaEngine invocation.

MediaAgent is optional for the Coordinator pipeline. If it is configured, this
node calls the real MediaEngine agent. If its API/search keys are missing,
imports fail, or execution times out, the node records a trace-only skip and
lets the rest of the Coordinator continue with the available agents.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from config import settings
from ..state import CoordinatorState, AgentRunResult
from ...utils.timeout_guard import with_timeout

MEDIA_AGENT_TIMEOUT = 180.0


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


async def _run_media_agent(query: str) -> Optional[str]:
    """Attempt to invoke MediaAgent. Returns Markdown string or None."""
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[4]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        if (settings.SEARCH_TOOL_TYPE or "").lower() == "anspireapi":
            from MediaEngine.agent import AnspireSearchAgent as MediaAgent
        else:
            from MediaEngine.agent import DeepSearchAgent as MediaAgent

        agent = MediaAgent()
        result = await agent.research_async(query)
        return result
    except Exception as exc:
        logger.info(f"[MediaAgentNode] MediaAgent skipped: {exc}")
        return None


async def media_agent_node(state: CoordinatorState) -> dict:
    """LangGraph node: execute MediaAgent when configured, otherwise skip quietly."""
    query = state["query"]
    logger.info(f"[MediaAgentNode] Starting for query: {query!r}")
    t0 = time.time()

    text_output: Optional[str] = None
    missing_config = _missing_media_config()

    if missing_config:
        mode = "skipped_unconfigured"
        detail = f"missing config: {', '.join(missing_config)}"
        logger.info(f"[MediaAgentNode] {mode} — {detail}")
    else:
        text_output = await with_timeout(
            _run_media_agent(query),
            timeout_seconds=MEDIA_AGENT_TIMEOUT,
            label="MediaAgent.research_async",
        )
        if text_output:
            mode = "live"
            detail = f"{len(text_output)} chars"
        else:
            mode = "skipped_unavailable"
            detail = "MediaAgent returned no output"

    duration = time.time() - t0

    run_result: AgentRunResult = {
        "agent_name": "media_agent",
        "success": True,
        "output": None,
        "text_output": text_output,
        "error": None,
        "duration_seconds": duration,
    }

    trace = f"[MediaAgentNode] {mode} — {detail} in {duration:.1f}s"
    logger.info(trace)

    return {
        "media_run": run_result,
        "coordinator_trace": [trace],
    }
