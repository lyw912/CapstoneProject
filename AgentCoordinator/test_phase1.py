"""
Phase 1 Integration Test for AgentCoordinator.

Tests the full pipeline end-to-end with:
- Real QueryAgent call (cached after first run)
- Injected test data for MediaAgent
- All Phase 1-4 nodes

Results are saved to AgentCoordinator/cache/test_results_YYYYMMDD_HHMMSS.json
and a report to AgentCoordinator/cache/test_report_YYYYMMDD_HHMMSS.md
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from loguru import logger


async def main():
    from AgentCoordinator.coordinator import AgentCoordinator

    query = "DeepSeek发布新模型 各方舆论"
    logger.info(f"=== AgentCoordinator Phase 1 Integration Test ===")
    logger.info(f"Query: {query!r}")

    coordinator = AgentCoordinator()
    t0 = time.time()

    try:
        result = await coordinator.run(query)
    except Exception as exc:
        logger.error(f"Pipeline FAILED: {exc}")
        raise

    duration = time.time() - t0

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    cache_dir = project_root / "AgentCoordinator" / "cache"
    cache_dir.mkdir(exist_ok=True)

    json_path = cache_dir / f"test_results_{ts}.json"
    report_path = cache_dir / f"test_report_{ts}.md"

    # Save JSON summary (exclude large report_output)
    summary = {k: v for k, v in result.items() if k not in ("report_output",)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save report
    report = result.get("report_output", "")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print test summary
    print("\n" + "=" * 70)
    print("AGENTCOORDINATOR PHASE 1 TEST RESULTS")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Total duration: {duration:.1f}s")
    print(f"Synthesis confidence: {result.get('synthesis_confidence', 0):.2f}")
    print(f"Agent errors: {result.get('agent_errors', [])}")
    print(f"\nDivergence hotspots ({len(result.get('divergence_hotspots', []))}):")
    for h in result.get("divergence_hotspots", []):
        print(f"  - {h}")
    print(f"\nConsensus points ({len(result.get('deliberation_consensus', []))}):")
    for c in result.get("deliberation_consensus", [])[:3]:
        print(f"  ✓ {c}")
    print(f"\nDissent points ({len(result.get('deliberation_dissents', []))}):")
    for d in result.get("deliberation_dissents", [])[:3]:
        print(f"  ✗ {d}")
    print(f"\nEcho warnings ({len(result.get('echo_warnings', []))}):")
    for w in result.get("echo_warnings", [])[:3]:
        print(f"  ⚠ {w}")
    print(f"\nVerified facts: {len(result.get('verified_facts', []))}")
    print(f"Platform interpretations: {list(result.get('platform_interpretations', {}).keys())}")
    print(f"\nCoordinator trace ({len(result.get('coordinator_trace', []))} entries):")
    for t in result.get("coordinator_trace", []):
        print(f"  {t}")
    print(f"\nReport length: {len(report)} chars")
    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {report_path}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    asyncio.run(main())
