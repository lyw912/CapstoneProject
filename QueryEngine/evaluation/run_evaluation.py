"""
Phase 2 Evaluation Execution Script

Supports three running modes:
  1. Quick validation (single query, default Q01):
       python -m QueryEngine.evaluation.run_evaluation --quick

  2. Full evaluation (20 queries, Exp-1 ~ Exp-4 ablation study):
       python -m QueryEngine.evaluation.run_evaluation --full

  3. Specify query IDs:
       python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16

Output:
  - Real-time metrics printed to console
  - Results saved to evaluation/results/YYYY-MM-DD_HH-MM.json

Reference: Architecture Doc v2.0 Part 3 § 12.4 ~ 12.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# Ensure project root is in sys.path
_here = Path(__file__).resolve()
_proj_root = _here.parents[2]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from QueryEngine.agent import DeepSearchAgent
from QueryEngine.evaluation.metrics import (
    compute_all_metrics,
    check_phase2_pass,
    format_metrics_report,
    PHASE2_TARGETS,
)
from QueryEngine.evaluation.test_queries import TEST_QUERIES, get_query_by_id

# Results output directory
_RESULTS_DIR = _here.parent / "results"


# ---------------------------------------------------------------------------
# Core Evaluation Functions
# ---------------------------------------------------------------------------

async def evaluate_single(
    agent: DeepSearchAgent,
    query_info: dict,
) -> dict:
    """Execute evaluation for a single query and return result dictionary."""
    query = query_info["query"]
    qid   = query_info["id"]
    logger.info(f"▶ [{qid}] Starting evaluation: {query!r}")

    start = time.time()
    try:
        output = await agent.research_structured(query)
        elapsed = time.time() - start

        metrics = compute_all_metrics(output, elapsed_seconds=elapsed)
        passed = check_phase2_pass(metrics)

        logger.info(f"  ✔ [{qid}] Completed, E2E={elapsed:.1f}s, SCS={metrics['scs']:.3f}, SDI={metrics['sdi']:.3f}")

        return {
            "id":      qid,
            "query":   query,
            "category": query_info.get("category", ""),
            "passed":  passed,
            "metrics": metrics,
            "output":  output,  # Save complete output
            "error":   None,
        }

    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"  ✖ [{qid}] Evaluation failed: {exc}")
        return {
            "id":      qid,
            "query":   query,
            "category": query_info.get("category", ""),
            "passed":  False,
            "metrics": {"scs": 0.0, "sdi": 0.0, "sbs": 0.0, "tsm": 0.0, "e2e": elapsed},
            "error":   str(exc),
        }


async def run_evaluation(
    query_infos: List[dict],
    save_results: bool = True,
) -> Dict:
    """
    Execute batch evaluation, running sequentially (to avoid concurrent LLM calls exceeding rate limits).

    Returns:
        Summary result dictionary
    """
    agent = DeepSearchAgent()
    results = []

    for qi in query_infos:
        result = await evaluate_single(agent, qi)
        results.append(result)

        # Real-time print
        print(f"\n{format_metrics_report(qi['query'], result['metrics'])}")
        print("-" * 60)

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------
    valid = [r for r in results if r["error"] is None]
    n = len(valid)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total":     len(results),
        "valid":     n,
        "passed":    sum(1 for r in valid if r["passed"]),
        "phase2_targets": PHASE2_TARGETS,
        "mean_metrics": {
            key: round(sum(r["metrics"].get(key, 0) for r in valid) / max(n, 1), 3)
            for key in ("scs", "sdi", "sbs", "tsm", "e2e")
        },
        "results": results,
    }

    # Whether overall passed
    m = summary["mean_metrics"]
    summary["overall_pass"] = (
        m["scs"] >= PHASE2_TARGETS["scs"]
        and m["sdi"] >= PHASE2_TARGETS["sdi"]
        and (m["e2e"] < 0 or m["e2e"] < PHASE2_TARGETS["e2e"])
    )

    # ------------------------------------------------------------------
    # Print Summary
    # ------------------------------------------------------------------
    overall = "✅ PHASE 2 PASS" if summary["overall_pass"] else "❌ PHASE 2 FAIL"
    print(f"\n{'='*60}")
    print(f"Evaluation Summary (Total {len(results)} items, Valid {n} items)")
    print(f"  Avg SCS={m['scs']:.3f}  Target≥{PHASE2_TARGETS['scs']}")
    print(f"  Avg SDI={m['sdi']:.3f}  Target≥{PHASE2_TARGETS['sdi']}")
    print(f"  Avg SBS={m['sbs']:.3f}  Target≥{PHASE2_TARGETS['sbs']}")
    print(f"  Avg TSM={m['tsm']:.3f}  Target≥{PHASE2_TARGETS['tsm']}")
    print(f"  Avg E2E={m['e2e']:.1f}s Target<{PHASE2_TARGETS['e2e']}s")
    print(f"\n{overall}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save Results
    # ------------------------------------------------------------------
    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_file = _RESULTS_DIR / f"eval_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {out_file}")

    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Agent Phase 2 Evaluation Script"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", action="store_true",
        help="Quick validation: run only Q01 (DeepSeek new model)",
    )
    group.add_argument(
        "--full", action="store_true",
        help="Full evaluation: run all 20 test queries",
    )
    group.add_argument(
        "--query", nargs="+", metavar="QID",
        help="Specify query IDs, e.g., --query Q01 Q06 Q16",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save evaluation result files",
    )
    return parser.parse_args()


async def _main():
    args = parse_args()

    if args.quick:
        query_infos = [get_query_by_id("Q01")]
    elif args.full:
        query_infos = TEST_QUERIES
    elif args.query:
        query_infos = [get_query_by_id(qid) for qid in args.query]
    else:
        # Default: quick validation
        query_infos = [get_query_by_id("Q01")]

    await run_evaluation(query_infos, save_results=not args.no_save)


if __name__ == "__main__":
    asyncio.run(_main())
