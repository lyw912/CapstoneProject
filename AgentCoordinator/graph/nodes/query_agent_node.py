"""
QueryAgentNode: Calls QueryEngine.DeepSearchAgent with timeout and result caching.

Caching: After the first successful run, results are saved to AgentCoordinator/cache/.
Subsequent runs load from cache to avoid repeated expensive API calls.
Cache is keyed by a hash of the query string.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from config import settings
from ..state import CoordinatorState, AgentRunResult
from ...utils.timeout_guard import run_sync_with_timeout

CACHE_DIR = Path(__file__).resolve().parents[3] / "AgentCoordinator" / "cache"


def _query_agent_timeout_seconds() -> float:
    return max(60.0, float(getattr(settings, "COORDINATOR_QUERY_AGENT_TIMEOUT", 3600) or 3600))


def _cache_path(query: str) -> Path:
    key = hashlib.md5(query.encode()).hexdigest()[:12]
    return CACHE_DIR / f"query_agent_{key}.json"


def _load_cache(query: str) -> Optional[Dict]:
    path = _cache_path(query)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[QueryAgentNode] Cache hit: {path.name}")
            return data
        except Exception as exc:
            logger.warning(f"[QueryAgentNode] Cache load failed: {exc}")
    return None


def _save_cache(query: str, output: Dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(query)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"[QueryAgentNode] Cached result → {path.name}")
    except Exception as exc:
        logger.warning(f"[QueryAgentNode] Cache save failed: {exc}")


def _run_query_agent_sync(query: str) -> Optional[Dict]:
    """Invoke DeepSearchAgent on a worker thread. Returns QueryAgentOutput dict or None."""
    try:
        import sys
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from QueryEngine.agent import DeepSearchAgent
        agent = DeepSearchAgent()
        return agent.research_structured_sync(query)
    except Exception as exc:
        logger.error(f"[QueryAgentNode] DeepSearchAgent raised: {exc}")
        return None


async def query_agent_node(state: CoordinatorState) -> dict:
    """LangGraph node: execute QueryAgent with caching and timeout."""
    query = state["query"]
    logger.info(f"[QueryAgentNode] Starting for query: {query!r}")
    t0 = time.time()

    # Try cache first
    cached = _load_cache(query)
    if cached is not None:
        run_result: AgentRunResult = {
            "agent_name": "query_agent",
            "success": True,
            "output": cached,
            "text_output": None,
            "error": None,
            "duration_seconds": 0.0,
        }
        trace = f"[QueryAgentNode] Loaded from cache in {time.time() - t0:.1f}s"
        logger.info(trace)
        return {"query_run": run_result, "coordinator_trace": [trace]}

    timeout_seconds = _query_agent_timeout_seconds()
    logger.info(f"[QueryAgentNode] Timeout budget: {timeout_seconds:.0f}s")

    # Run the LangGraph subgraph in a worker thread so Media/Query can execute
    # in parallel without blocking the asyncio event loop (which breaks wait_for).
    output = await run_sync_with_timeout(
        _run_query_agent_sync,
        timeout_seconds,
        query,
        label="QueryAgent.research_structured",
    )

    duration = time.time() - t0

    if output is not None:
        _save_cache(query, output)
        # Also set analysis_type from output
        analysis_type = output.get("analysis_type", "general")
        run_result = {
            "agent_name": "query_agent",
            "success": True,
            "output": output,
            "text_output": None,
            "error": None,
            "duration_seconds": duration,
        }
        trace = (
            f"[QueryAgentNode] Success in {duration:.1f}s — "
            f"sources={output.get('total_sources_kept', 0)}, "
            f"coverage={output.get('coverage_score', 0):.2f}, "
            f"analysis_type={analysis_type}"
        )
        logger.info(trace)
        return {
            "query_run": run_result,
            "analysis_type": analysis_type,
            "coordinator_trace": [trace],
        }
    else:
        error_msg = f"QueryAgent failed or timed out after {duration:.1f}s"
        run_result = {
            "agent_name": "query_agent",
            "success": False,
            "output": None,
            "text_output": None,
            "error": error_msg,
            "duration_seconds": duration,
        }
        trace = f"[QueryAgentNode] FAILED: {error_msg}"
        logger.error(trace)
        return {
            "query_run": run_result,
            "agent_errors": [error_msg],
            "coordinator_trace": [trace],
        }
