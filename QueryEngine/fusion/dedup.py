"""
MinHash LSH content deduplication

Using datasketch library to implement MinHash + LSH deduplication:
  - 3-gram shingling (character-level, compatible with Chinese and English)
  - 128 permutation functions (balance between accuracy and speed)
  - Default threshold 0.8 (80% Jaccard similarity considered duplicate)

If datasketch is not installed, automatically degrade to no-op (pass-through input) and issue a warning.

Reference: Architecture Document v2.0 Part 2 § 8.3
"""

from __future__ import annotations

from typing import List

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency: datasketch
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash, MinHashLSH
    _MINHASH_AVAILABLE = True
except ImportError:
    _MINHASH_AVAILABLE = False
    logger.warning(
        "[MinHashDedup] datasketch not installed, MinHash content deduplication will be skipped."
        " Please run: pip install datasketch"
    )

# ---------------------------------------------------------------------------
# Minimum text length: snippets that are too short cannot reliably compute shingling
# ---------------------------------------------------------------------------
_MIN_TEXT_LEN: int = 15


def minhash_dedup(
    sources: List[dict],
    threshold: float = 0.80,
    num_perm: int = 128,
) -> List[dict]:
    """
    Content deduplication based on MinHash LSH.

    Perform 3-gram shingling on the snippet of each source in sources,
    if Jaccard similarity between two sources ≥ threshold, they are considered duplicates,
    and the one that appeared first is kept.

    Args:
        sources:    List of SourceItem dicts (processed in insertion order)
        threshold:  Jaccard similarity threshold, [0, 1], default 0.80
        num_perm:   Number of MinHash permutations (higher is more accurate but slower), default 128

    Returns:
        Deduplicated list of SourceItem, maintaining original relative order
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

        # Text too short -> cannot reliably deduplicate, keep directly
        if len(text) < _MIN_TEXT_LEN:
            unique.append(source)
            continue

        # Skip when source_id is empty (should not happen, but defensive handling)
        if not source_id:
            unique.append(source)
            continue

        # Build MinHash (3-gram shingling, character-level)
        mh = MinHash(num_perm=num_perm)
        for i in range(max(len(text) - 2, 1)):
            mh.update(text[i:i + 3].encode("utf-8"))

        try:
            similar_ids = lsh.query(mh)
            if not similar_ids:
                # No similar items -> new document, insert into LSH and keep
                if source_id not in inserted_ids:
                    lsh.insert(source_id, mh)
                    inserted_ids.add(source_id)
                unique.append(source)
            # else: similar to existing sources -> discard (duplicate)

        except Exception as exc:
            # Conservatively keep when datasketch internal error occurs
            logger.debug(f"[MinHashDedup] Skipping {source_id}: {exc}")
            unique.append(source)

    return unique
