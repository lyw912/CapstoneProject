"""Run the live Coordinator-to-ReportEngine validation path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AgentCoordinator import AgentCoordinator
from AgentCoordinator.utils.report_bridge import generate_report_engine_html


LATEST_ARTIFACT = PROJECT_ROOT / "AgentCoordinator" / "cache" / "coordinator_output_latest.json"
VALIDATION_DIR = PROJECT_ROOT / "output" / "e2e_validation"


def _progress(node: str, update: Dict[str, Any], _state: Dict[str, Any], elapsed: float) -> None:
    counts = {
        key: value
        for key, value in update.items()
        if key in {"completed_agents", "evidence_graph_summary", "tasks"}
    }
    print(json.dumps({"stage": node, "elapsed_seconds": round(elapsed, 2), **counts}, ensure_ascii=False), flush=True)


def _artifact_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    intelligence = artifact.get("coordinator_intelligence") or {}
    graph = intelligence.get("evidence_graph") or {}
    source_data = artifact.get("source_data") or {}
    query_data = source_data.get("query_agent") or {}
    media_data = source_data.get("media_agent") or {}
    diagnostics = intelligence.get("provider_diagnostics") or []
    return {
        "schema_version": artifact.get("schema_version"),
        "query": artifact.get("query"),
        "run_id": intelligence.get("run_id"),
        "normalized_items": len(graph.get("normalized_items") or []),
        "claims": len(graph.get("claims") or []),
        "audit_decisions": len(graph.get("audit_decisions") or []),
        "query_sources": query_data.get("total_sources", 0),
        "mindspider_mode": (query_data.get("social_sentiment") or {}).get("mode"),
        "mindspider_posts": (query_data.get("social_sentiment") or {}).get("total_posts", 0),
        "media_available": bool(media_data.get("available")),
        "media_dossiers": media_data.get("section_dossiers", 0),
        "provider_diagnostics": diagnostics,
        "agent_errors": artifact.get("agent_errors") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.report_only:
        artifact = json.loads(LATEST_ARTIFACT.read_text(encoding="utf-8"))
        if artifact.get("query") != args.query:
            raise SystemExit(
                f"Latest artifact query {artifact.get('query')!r} does not match {args.query!r}"
            )
        summary = {
            "status": "coordinator_reused",
            "coordinator_seconds": 0.0,
            "coordinator_output_path": str(LATEST_ARTIFACT),
            "latest_artifact": str(LATEST_ARTIFACT),
            "artifact": _artifact_summary(artifact),
        }
    else:
        coordinator = AgentCoordinator(use_checkpointing=False)
        result = coordinator.run_sync(args.query, progress_callback=_progress)
        artifact = json.loads(LATEST_ARTIFACT.read_text(encoding="utf-8"))
        summary = {
            "status": "coordinator_complete",
            "coordinator_seconds": round(time.time() - started, 2),
            "coordinator_output_path": result.get("coordinator_output_path"),
            "latest_artifact": str(LATEST_ARTIFACT),
            "artifact": _artifact_summary(artifact),
        }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = VALIDATION_DIR / f"validation_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_report:
        report_started = time.time()
        try:
            report = generate_report_engine_html(artifact, save_report=True)
            summary.update(
                {
                    "status": "complete",
                    "report_seconds": round(time.time() - report_started, 2),
                    "report_id": report.get("report_id"),
                    "html_length": len(report.get("html_content") or ""),
                    "report_outputs": {
                        key: value
                        for key, value in report.items()
                        if key.endswith("_path") or key in {"report_id"}
                    },
                }
            )
        except Exception as exc:
            summary.update(
                {
                    "status": "report_failed",
                    "report_seconds": round(time.time() - report_started, 2),
                    "report_error": f"{type(exc).__name__}: {exc}",
                }
            )
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return 1

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
