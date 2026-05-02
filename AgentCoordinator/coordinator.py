"""
AgentCoordinator: Unified entry point.

Usage:
    from AgentCoordinator.coordinator import AgentCoordinator

    coordinator = AgentCoordinator()

    # Async
    result = await coordinator.run("DeepSeek发布新模型 各方舆论")

    # Sync
    result = coordinator.run_sync("DeepSeek发布新模型 各方舆论")
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class AgentCoordinator:
    """Main coordinator class that runs the full multi-agent analysis pipeline."""

    def __init__(self, use_checkpointing: bool = True):
        """
        Initialize the AgentCoordinator.

        Args:
            use_checkpointing: If True (default), compile the LangGraph with a
                MemorySaver checkpointer so each run can be resumed on failure
                using its unique thread_id.
        """
        self.use_checkpointing = use_checkpointing
        self._graph = None  # Lazy-initialized

    @property
    def graph(self):
        if self._graph is None:
            from .graph.builder import build_coordinator_graph
            self._graph = build_coordinator_graph(use_checkpointing=self.use_checkpointing)
            logger.info("[AgentCoordinator] LangGraph compiled successfully")
        return self._graph

    def _build_initial_state(self, query: str) -> Dict:
        return {
            "query": query,
            "analysis_type": "general",       # May be overwritten by query_agent_node
            "query_run": None,
            "media_run": None,
            "agent_errors": [],
            "bridged_propositions": None,
            "divergence_matrix": None,
            "divergence_hotspots": None,
            "perspectives": None,
            "deliberation_rounds": None,
            "deliberation_consensus": None,
            "deliberation_dissents": None,
            "search_gaps": None,
            "supplementary_results": None,
            "search_rounds": 0,
            "echo_warnings": None,
            "silent_majority_hypothesis": None,
            "verified_facts": None,
            "opinions_sentiments": None,
            "analytical_frameworks": None,
            "platform_interpretations": None,
            "synthesis_context": None,
            "synthesis_confidence": 0.0,
            "report_output": None,
            "coordinator_trace": [],
        }

    async def run(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full coordinator pipeline asynchronously.

        Args:
            query: The analysis query string.
            thread_id: Optional checkpoint thread identifier. If None, a random
                UUID is generated so each run gets an isolated checkpoint scope.
                Pass the same thread_id to resume a failed run from its last
                successful node.

        Returns a dict with:
          - report_output: Final report (HTML or Markdown)
          - synthesis_context: Structured synthesis data
          - coordinator_trace: Full execution trace
          - duration_seconds: Total elapsed time
          - coordinator_output_path: Path to the saved coordinator_output.json
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        logger.info(f"[AgentCoordinator] Starting pipeline for: {query!r}  (thread_id={thread_id})")
        t0 = time.time()

        initial_state = self._build_initial_state(query)

        # Build invoke config — only pass configurable if checkpointing is on
        invoke_config: Dict[str, Any] = {}
        if self.use_checkpointing:
            invoke_config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config=invoke_config if invoke_config else None)
        except Exception as exc:
            logger.error(f"[AgentCoordinator] Pipeline failed: {exc}")
            raise

        duration = time.time() - t0
        logger.info(f"[AgentCoordinator] Pipeline complete in {duration:.1f}s")

        result = {
            "query": query,
            "thread_id": thread_id,
            "report_output": final_state.get("report_output", ""),
            "synthesis_context": final_state.get("synthesis_context", {}),
            "synthesis_confidence": final_state.get("synthesis_confidence", 0.0),
            "divergence_matrix": final_state.get("divergence_matrix", {}),
            "divergence_hotspots": final_state.get("divergence_hotspots", []),
            "deliberation_consensus": final_state.get("deliberation_consensus", []),
            "deliberation_dissents": final_state.get("deliberation_dissents", []),
            "echo_warnings": final_state.get("echo_warnings", []),
            "verified_facts": final_state.get("verified_facts", []),
            "platform_interpretations": final_state.get("platform_interpretations", {}),
            "coordinator_trace": final_state.get("coordinator_trace", []),
            "agent_errors": final_state.get("agent_errors", []),
            "duration_seconds": duration,
        }

        # Export clean coordinator_output.json for ReportAgent consumption
        coordinator_output_path = self._export_coordinator_output(result, query)
        result["coordinator_output_path"] = coordinator_output_path

        return result

    def _export_coordinator_output(self, result: Dict[str, Any], query: str) -> str:
        """
        Build and save the clean coordinator_output.json artifact.

        Saves two files:
          - coordinator_output_{YYYYMMDD_HHMMSS}.json  (timestamped archive)
          - coordinator_output_latest.json              (always overwritten, for ReportAgent)

        Returns the path to the timestamped file.
        """
        from .coordinator_output_schema import build_coordinator_output

        ts = time.strftime("%Y%m%d_%H%M%S")
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)

        structured = build_coordinator_output(
            result=result,
            query=query,
            duration_seconds=result.get("duration_seconds", 0.0),
        )

        # Timestamped archive
        timestamped_path = cache_dir / f"coordinator_output_{ts}.json"
        with open(timestamped_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)

        # Fixed "latest" path for ReportAgent
        latest_path = cache_dir / "coordinator_output_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)

        logger.info(f"[AgentCoordinator] coordinator_output saved → {timestamped_path}")
        logger.info(f"[AgentCoordinator] coordinator_output_latest.json updated → {latest_path}")

        return str(timestamped_path)

    def run_sync(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for run()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(query, thread_id=thread_id))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(query, thread_id=thread_id))
        except RuntimeError:
            return asyncio.run(self.run(query, thread_id=thread_id))

    def save_result(self, result: Dict, output_path: Optional[str] = None) -> str:
        """Save coordinator result to a JSON log file and report file."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(__file__).parent / "cache"
        base.mkdir(exist_ok=True)

        if output_path is None:
            output_path = str(base / f"coordinator_result_{ts}.json")

        # Save JSON (excluding the full report text for cleanliness)
        json_data = {k: v for k, v in result.items() if k != "report_output"}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # Save report separately
        report = result.get("report_output", "")
        if report:
            report_path = output_path.replace(".json", "_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"[AgentCoordinator] Report saved → {report_path}")

        logger.info(f"[AgentCoordinator] Result saved → {output_path}")
        return output_path
