"""
QueryEngine fusion module

- rrf_fuse:      Reciprocal Rank Fusion multi-source result fusion (SIGIR 2009)
- minhash_dedup: MinHash LSH content deduplication (datasketch)
"""

from .rrf import rrf_fuse
from .dedup import minhash_dedup

__all__ = [
    "rrf_fuse",
    "minhash_dedup",
]
