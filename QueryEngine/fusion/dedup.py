"""
MinHash LSH 内容去重

使用 datasketch 库实现 MinHash + LSH 去重：
  - 3-gram shingling（字符级，兼容中英文）
  - 128 置换函数（精度与速度的平衡）
  - 默认阈值 0.8（80% Jaccard 相似度视为重复）

若 datasketch 未安装，自动降级为无操作（透传输入）并发出警告。

参考：架构文档 v2.0 Part 2 § 8.3
"""

from __future__ import annotations

from typing import List

from loguru import logger

# ---------------------------------------------------------------------------
# 可选依赖：datasketch
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash, MinHashLSH
    _MINHASH_AVAILABLE = True
except ImportError:
    _MINHASH_AVAILABLE = False
    logger.warning(
        "[MinHashDedup] datasketch 未安装，MinHash 内容去重将跳过。"
        " 请执行: pip install datasketch"
    )

# ---------------------------------------------------------------------------
# 最短文本长度：太短的 snippet 无法可靠计算 shingling
# ---------------------------------------------------------------------------
_MIN_TEXT_LEN: int = 15


def minhash_dedup(
    sources: List[dict],
    threshold: float = 0.80,
    num_perm: int = 128,
) -> List[dict]:
    """
    基于 MinHash LSH 的内容去重。

    对 sources 中每条来源的 snippet 做 3-gram shingling，
    若两条来源的 Jaccard 相似度 ≥ threshold，则视为重复，保留先出现的一条。

    Args:
        sources:    SourceItem 字典列表（按插入顺序处理）
        threshold:  Jaccard 相似度阈值，[0, 1]，默认 0.80
        num_perm:   MinHash 置换数（越高越精确但越慢），默认 128

    Returns:
        去重后的 SourceItem 列表，保持原相对顺序
    """
    if not _MINHASH_AVAILABLE:
        return sources

    if not sources:
        return sources

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique: List[dict] = []
    inserted_ids: set[str] = set()

    for source in sources:
        text: str = source.get("snippet") or ""
        source_id: str = source.get("source_id") or ""

        # 文本太短 → 无法可靠去重，直接保留
        if len(text) < _MIN_TEXT_LEN:
            unique.append(source)
            continue

        # source_id 为空时跳过（不应发生，但防御性处理）
        if not source_id:
            unique.append(source)
            continue

        # 构建 MinHash（3-gram shingling，字符级）
        mh = MinHash(num_perm=num_perm)
        for i in range(max(len(text) - 2, 1)):
            mh.update(text[i:i + 3].encode("utf-8"))

        try:
            similar_ids = lsh.query(mh)
            if not similar_ids:
                # 无相似项 → 新文档，插入 LSH 并保留
                if source_id not in inserted_ids:
                    lsh.insert(source_id, mh)
                    inserted_ids.add(source_id)
                unique.append(source)
            # else: 与已有来源相似 → 丢弃（重复）

        except Exception as exc:
            # datasketch 内部报错时保守保留
            logger.debug(f"[MinHashDedup] 跳过 {source_id}: {exc}")
            unique.append(source)

    return unique
