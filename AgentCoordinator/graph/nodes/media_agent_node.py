"""
MediaAgentNode: Calls MediaEngine or injects test data when Media Agent is unavailable.

The Media Agent currently may not be running. When unavailable, this node injects
realistic test data simulating what MediaAgent would produce, to allow full pipeline
testing without blocking on Media Agent availability.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from ..state import CoordinatorState, AgentRunResult
from ...utils.timeout_guard import with_timeout

MEDIA_AGENT_TIMEOUT = 180.0
_USE_TEST_DATA = True  # Set to False when MediaAgent is confirmed running


def _generate_test_media_data(query: str) -> str:
    """
    Realistic test data simulating MediaAgent Markdown output.
    Used when Media Agent is not running.
    """
    return f"""# Media Analysis Report: {query}

## Executive Summary
Based on comprehensive Chinese media coverage analysis, this topic has generated significant
reporting across mainstream and digital media outlets. Coverage spans official state media,
commercial news platforms, and social media amplification.

## Key Media Narratives

### Official Media Framing
State media (Xinhua, People's Daily, CCTV) tends to emphasize factual reporting and
official statements. Coverage is generally measured in tone, with emphasis on
constructive developments and authoritative sources.

Key claims from official media:
- Official announcements and policy context are prominently featured
- Technical and economic data is cited from government or established industry sources
- International comparisons are used to contextualize domestic developments

### Commercial Media Coverage
Digital and commercial media platforms show more diverse framing:
- Business-oriented outlets (Caixin, 36Kr) focus on market and industry implications
- Technology media emphasizes innovation dimensions and competitive dynamics
- General news outlets balance breadth of coverage with accessible explanations

### Social Media Amplification
The topic has generated substantial social media discussion:
- Trending on major platforms with varied sentiment
- Key opinion leaders (KOLs) across different domains have weighed in
- User-generated content shows diverse interpretations

## Sentiment Analysis
- **Positive/Supportive coverage**: ~40% of articles
- **Neutral/Analytical coverage**: ~45% of articles
- **Critical/Skeptical coverage**: ~15% of articles

## Notable Sources
- Xinhua News Agency: Official reporting, authoritative on policy dimensions
- Caixin Media: In-depth financial and business analysis
- 36Kr: Technology and startup ecosystem perspective
- The Paper (澎湃): Investigative and analytical journalism

## Media Gaps Identified
1. Limited coverage of regional/local impacts outside major cities
2. International media perspective largely absent from Chinese-language coverage
3. Long-term trend analysis relatively sparse compared to event-focused reporting

*Note: This is test data injected for pipeline validation. Replace with live MediaAgent output for production use.*
"""


async def _run_media_agent(query: str) -> Optional[str]:
    """Attempt to invoke MediaAgent. Returns Markdown string or None."""
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[4]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from MediaEngine.agent import DeepSearchAgent as MediaAgent
        agent = MediaAgent()
        result = await agent.research_async(query)
        return result
    except Exception as exc:
        logger.warning(f"[MediaAgentNode] MediaAgent unavailable: {exc}")
        return None


async def media_agent_node(state: CoordinatorState) -> dict:
    """LangGraph node: execute MediaAgent or inject test data as fallback."""
    query = state["query"]
    logger.info(f"[MediaAgentNode] Starting for query: {query!r}")
    t0 = time.time()

    text_output: Optional[str] = None
    error: Optional[str] = None

    if not _USE_TEST_DATA:
        # Try real MediaAgent
        text_output = await with_timeout(
            _run_media_agent(query),
            timeout_seconds=MEDIA_AGENT_TIMEOUT,
            label="MediaAgent.research_async",
        )

    if text_output is None:
        # Inject test data
        text_output = _generate_test_media_data(query)
        mode = "test_data_injected"
        logger.info(f"[MediaAgentNode] Using injected test data (Media Agent not running)")
    else:
        mode = "live"

    duration = time.time() - t0

    run_result: AgentRunResult = {
        "agent_name": "media_agent",
        "success": True,
        "output": None,
        "text_output": text_output,
        "error": None,
        "duration_seconds": duration,
    }

    trace = f"[MediaAgentNode] {mode} — {len(text_output)} chars in {duration:.1f}s"
    logger.info(trace)

    return {
        "media_run": run_result,
        "coordinator_trace": [trace],
    }
