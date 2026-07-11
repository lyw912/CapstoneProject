"""AgentCoordinator entry point.

External callers still import ``AgentCoordinator`` and still receive a
coordinator_output_latest.json artifact. Internally, the active path uses the
Coordinator intelligence layer as shared evidence state, then exports the
legacy Coordinator fields as views over that state.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger


class AgentCoordinator:
    """Main coordinator class for the active evidence-audited analysis path."""

    def __init__(self, use_checkpointing: bool = True):
        self.use_checkpointing = use_checkpointing
        self._graph = None

    @property
    def graph(self):
        raise RuntimeError(
            "The active AgentCoordinator path uses the internal intelligence layer. "
            "The legacy LangGraph nodes remain in AgentCoordinator/graph for compatibility and reference."
        )

    async def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any], Dict[str, Any], float], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the Coordinator intelligence layer asynchronously."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        from .intelligence import CoordinatorIntelligenceLayer, CoordinatorIntelligenceRequest

        logger.info("[AgentCoordinator] Starting intelligence layer for: {!r} (thread_id={})", query, thread_id)
        started = time.time()
        layer = CoordinatorIntelligenceLayer()
        artifact = layer.run(
            CoordinatorIntelligenceRequest(query=query, thread_id=thread_id),
            progress_callback=progress_callback,
        )
        duration = time.time() - started
        output = self._export_coordinator_output(artifact, duration_seconds=duration)
        synthesis = output.get("synthesis") or {}
        result = {
            "query": artifact.query,
            "thread_id": thread_id,
            "report_output": artifact.synthesis_markdown,
            "coordinator_intelligence": artifact.to_dict(),
            "synthesis_context": {
                "coordinator_intelligence": artifact.to_dict(),
                "top_insights": synthesis.get("top_insights", []),
                "key_tensions": synthesis.get("key_tensions", []),
                "overall_confidence": synthesis.get("overall_confidence", 0.0),
                "synthesis_summary": synthesis.get("summary", ""),
            },
            "synthesis_confidence": synthesis.get("overall_confidence", 0.0),
            "divergence_matrix": output.get("divergence_matrix", {}),
            "divergence_hotspots": (output.get("divergence_matrix") or {}).get("hotspots", []),
            "deliberation_consensus": (output.get("deliberation") or {}).get("final_consensus", []),
            "deliberation_dissents": (output.get("deliberation") or {}).get("final_dissents", []),
            "echo_warnings": (output.get("bias_analysis") or {}).get("echo_warnings", []),
            "verified_facts": (output.get("fact_opinion_separation") or {}).get("verified_facts", []),
            "platform_interpretations": output.get("platform_interpretations", {}),
            "coordinator_trace": output.get("coordinator_trace", []),
            "agent_errors": output.get("agent_errors", []),
            "duration_seconds": duration,
            "coordinator_output_path": output.get("_coordinator_output_path", ""),
        }
        logger.info("[AgentCoordinator] intelligence layer complete in {:.1f}s", duration)
        return result

    def _export_coordinator_output(self, artifact, duration_seconds: float) -> Dict[str, Any]:
        from .intelligence.projection import build_coordinator_output_from_artifact

        ts = time.strftime("%Y%m%d_%H%M%S")
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)

        structured = build_coordinator_output_from_artifact(artifact, duration_seconds=duration_seconds)

        timestamped_path = cache_dir / f"coordinator_output_{ts}.json"
        with open(timestamped_path, "w", encoding="utf-8") as handle:
            json.dump(structured, handle, ensure_ascii=False, indent=2)

        latest_path = cache_dir / "coordinator_output_latest.json"
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(structured, handle, ensure_ascii=False, indent=2)

        structured["_coordinator_output_path"] = str(timestamped_path)
        logger.info("[AgentCoordinator] coordinator_output saved -> {}", timestamped_path)
        logger.info("[AgentCoordinator] coordinator_output_latest.json updated -> {}", latest_path)
        return structured

    def run_sync(
        self,
        query: str,
        thread_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any], Dict[str, Any], float], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for run()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.run(query, thread_id=thread_id, progress_callback=progress_callback),
                    )
                    return future.result()
            return loop.run_until_complete(self.run(query, thread_id=thread_id, progress_callback=progress_callback))
        except RuntimeError:
            return asyncio.run(self.run(query, thread_id=thread_id, progress_callback=progress_callback))

    def save_result(self, result: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Save a coordinator result to disk for manual inspection."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(__file__).parent / "cache"
        base.mkdir(exist_ok=True)
        if output_path is None:
            output_path = str(base / f"coordinator_result_{ts}.json")
        json_data = {key: value for key, value in result.items() if key != "report_output"}
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(json_data, handle, ensure_ascii=False, indent=2)
        report = result.get("report_output", "")
        if report:
            report_path = output_path.replace(".json", "_report.md")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(report)
            logger.info("[AgentCoordinator] Report saved -> {}", report_path)
        logger.info("[AgentCoordinator] Result saved -> {}", output_path)
        return output_path
