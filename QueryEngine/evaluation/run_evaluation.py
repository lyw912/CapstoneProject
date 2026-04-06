"""
Phase 2 评估执行脚本

支持两种运行模式：
  1. 快速验证（单条查询，默认 Q01）：
       python -m QueryEngine.evaluation.run_evaluation --quick

  2. 完整评估（20条查询，Exp-1 ~ Exp-4 消融实验）：
       python -m QueryEngine.evaluation.run_evaluation --full

  3. 指定查询 ID：
       python -m QueryEngine.evaluation.run_evaluation --query Q01 Q06 Q16

输出：
  - 控制台实时打印各指标
  - 结果保存至 evaluation/results/YYYY-MM-DD_HH-MM.json

参考：架构文档 v2.0 Part 3 § 12.4 ~ 12.5
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

# 确保项目根目录在 sys.path
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

# 结果输出目录
_RESULTS_DIR = _here.parent / "results"


# ---------------------------------------------------------------------------
# 核心评估函数
# ---------------------------------------------------------------------------

async def evaluate_single(
    agent: DeepSearchAgent,
    query_info: dict,
) -> dict:
    """对单条查询执行评估并返回结果字典。"""
    query = query_info["query"]
    qid   = query_info["id"]
    logger.info(f"▶ [{qid}] 开始评估: {query!r}")

    start = time.time()
    try:
        output = await agent.research_structured(query)
        elapsed = time.time() - start

        metrics = compute_all_metrics(output, elapsed_seconds=elapsed)
        passed = check_phase2_pass(metrics)

        logger.info(f"  ✔ [{qid}] 完成, E2E={elapsed:.1f}s, SCS={metrics['scs']:.3f}, SDI={metrics['sdi']:.3f}")

        return {
            "id":      qid,
            "query":   query,
            "category": query_info.get("category", ""),
            "passed":  passed,
            "metrics": metrics,
            "output":  output,  # 保存完整输出
            "error":   None,
        }

    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"  ✖ [{qid}] 评估失败: {exc}")
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
    执行批量评估，顺序运行（避免并发 LLM 调用超出限速）。

    Returns:
        汇总结果字典
    """
    agent = DeepSearchAgent()
    results = []

    for qi in query_infos:
        result = await evaluate_single(agent, qi)
        results.append(result)

        # 实时打印
        print(f"\n{format_metrics_report(qi['query'], result['metrics'])}")
        print("-" * 60)

    # ------------------------------------------------------------------
    # 汇总统计
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

    # 是否整体通过
    m = summary["mean_metrics"]
    summary["overall_pass"] = (
        m["scs"] >= PHASE2_TARGETS["scs"]
        and m["sdi"] >= PHASE2_TARGETS["sdi"]
        and (m["e2e"] < 0 or m["e2e"] < PHASE2_TARGETS["e2e"])
    )

    # ------------------------------------------------------------------
    # 打印汇总
    # ------------------------------------------------------------------
    overall = "✅ PHASE 2 PASS" if summary["overall_pass"] else "❌ PHASE 2 FAIL"
    print(f"\n{'='*60}")
    print(f"评估汇总（共 {len(results)} 条，有效 {n} 条）")
    print(f"  平均 SCS={m['scs']:.3f}  目标≥{PHASE2_TARGETS['scs']}")
    print(f"  平均 SDI={m['sdi']:.3f}  目标≥{PHASE2_TARGETS['sdi']}")
    print(f"  平均 SBS={m['sbs']:.3f}  目标≥{PHASE2_TARGETS['sbs']}")
    print(f"  平均 TSM={m['tsm']:.3f}  目标≥{PHASE2_TARGETS['tsm']}")
    print(f"  平均 E2E={m['e2e']:.1f}s 目标<{PHASE2_TARGETS['e2e']}s")
    print(f"\n{overall}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_file = _RESULTS_DIR / f"eval_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存至: {out_file}")

    return summary


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Agent Phase 2 评估脚本"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--quick", action="store_true",
        help="快速验证：仅运行 Q01（DeepSeek新模型）",
    )
    group.add_argument(
        "--full", action="store_true",
        help="完整评估：运行全部 20 条测试查询",
    )
    group.add_argument(
        "--query", nargs="+", metavar="QID",
        help="指定查询 ID，如 --query Q01 Q06 Q16",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="不保存评估结果文件",
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
        # 默认：快速验证
        query_infos = [get_query_by_id("Q01")]

    await run_evaluation(query_infos, save_results=not args.no_save)


if __name__ == "__main__":
    asyncio.run(_main())
