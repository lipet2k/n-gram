"""Build `table.0`, the suffix array of a `tokenized.0` file, in infini-gram's format."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pydivsufsort
from loguru import logger

from src.config import IndexFormat


def pointer_size(ds_size: int) -> int:
    """Bytes per table entry, `ceil(log2(ds_size) / 8)`, as infini-gram's indexer chooses it."""
    return max(1, math.ceil(math.log2(ds_size) / 8))


def build_table(
    index_dir: Path | str, fmt: IndexFormat, write_chunk: int
) -> tuple[int, int]:
    """Write `table.0` next to `tokenized.0`. Returns (number of ranks, pointer size)."""
    index_dir = Path(index_dir)
    ds = np.fromfile(index_dir / "tokenized.0", dtype=np.uint8)
    logger.info("sorting {} byte suffixes", len(ds))
    sa = pydivsufsort.divsufsort(ds)
    del ds
    sa = sa[(sa % fmt.token_bytes) == 0]
    ptr_size = pointer_size(len(sa) * fmt.token_bytes)
    with open(index_dir / "table.0", "wb") as out:
        for start in range(0, len(sa), write_chunk):
            block = (
                sa[start : start + write_chunk]
                .astype("<u8")
                .view(np.uint8)
                .reshape(-1, 8)
            )
            out.write(np.ascontiguousarray(block[:, :ptr_size]).tobytes())
    logger.info(
        "wrote table.0 with {} ranks at {} bytes per pointer", len(sa), ptr_size
    )
    return len(sa), ptr_size


def brute_force_interval(ds: bytes, query: bytes, token_bytes: int) -> tuple[int, int]:
    """Reference for tests: the rank interval of `query` from sorting every aligned suffix."""
    suffixes = sorted(ds[i:] for i in range(0, len(ds), token_bytes))
    lo = sum(1 for s in suffixes if s < query)
    count = sum(1 for s in suffixes if s[: len(query)] == query)
    return lo, lo + count
