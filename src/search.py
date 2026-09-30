"""Suffix-array searches over a Store.

`find` mirrors infini-gram's `_find_thread` step for step: one binary search for any
matching rank, then one for each boundary inside the remaining halves. `ngram_runs` and
`children` enumerate the distinct n-grams of a rank range with their intervals, which is
how caches are built.
"""

from __future__ import annotations

import numpy as np

from src.store import MemoryStore, Store


def find(
    store: Store, query: bytes, lo: int = 0, hi: int | None = None
) -> tuple[int, int]:
    """Rank interval of the suffixes starting with `query`, searched inside [lo, hi)."""
    if hi is None:
        hi = store.num_ranks
    nbytes = len(query)
    if nbytes == 0:
        return lo, hi
    mi = lo
    while lo < hi:
        mi = (lo + hi - 1) >> 1
        suffix = store.suffix(mi, nbytes)
        if suffix < query:
            lo = mi + 1
        elif suffix > query:
            hi = mi
        else:
            break
    if lo == hi:
        return lo, lo
    left, right = lo - 1, mi
    while right - left > 1:
        m = (left + right) >> 1
        if store.suffix(m, nbytes) < query:
            left = m
        else:
            right = m
    start = right
    left, right = mi, hi
    while right - left > 1:
        m = (left + right) >> 1
        if query < store.suffix(m, nbytes):
            right = m
        else:
            left = m
    return start, right


def ngram_runs(
    store: MemoryStore, lo: int, hi: int, depth: int, k: int, chunk: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct k-token sequences found `depth` tokens into the suffixes of ranks [lo, hi).

    Returns (keys, los, his): keys is (m, k) in rank order and [los[i], his[i]) is the
    rank interval of key i. Sequences that run past the end of the file or contain the
    document separator are left out. Ranks are decoded `chunk` at a time, the store's
    chunk size unless given.
    """
    chunk = store.chunk_ranks if chunk is None else chunk
    keys: list[np.ndarray] = []
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    open_key, open_valid, open_lo = None, False, lo
    for start in range(lo, hi, chunk):
        end = min(start + chunk, hi)
        rows, valid = _ngram_rows(store, start, end, depth, k)
        change = np.ones(end - start, dtype=bool)
        change[1:] = np.any(rows[1:] != rows[:-1], axis=1) | (valid[1:] != valid[:-1])
        if (
            open_key is not None
            and valid[0] == open_valid
            and np.array_equal(rows[0], open_key)
        ):
            change[0] = False
        starts = np.flatnonzero(change)
        if len(starts) == 0:
            continue
        if open_key is not None:
            _emit(
                keys,
                los,
                his,
                open_key[None],
                [open_valid],
                [open_lo],
                [start + starts[0]],
            )
        _emit(
            keys,
            los,
            his,
            rows[starts[:-1]],
            valid[starts[:-1]],
            start + starts[:-1],
            start + starts[1:],
        )
        last = starts[-1]
        open_key, open_valid, open_lo = (
            rows[last].copy(),
            bool(valid[last]),
            start + last,
        )
    if open_key is not None:
        _emit(keys, los, his, open_key[None], [open_valid], [open_lo], [hi])
    if not keys:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, k), dtype=store.tokens.dtype), empty, empty
    return np.concatenate(keys), np.concatenate(los), np.concatenate(his)


def _emit(keys, los, his, rows, valid, lo, hi) -> None:
    valid = np.asarray(valid, dtype=bool)
    if np.any(valid):
        keys.append(np.asarray(rows)[valid])
        los.append(np.asarray(lo, dtype=np.int64)[valid])
        his.append(np.asarray(hi, dtype=np.int64)[valid])


def _ngram_rows(store: MemoryStore, start: int, end: int, depth: int, k: int):
    first = store.ptrs(start, end) // store.fmt.token_bytes + depth
    idx = first[:, None] + np.arange(k, dtype=np.int64)
    valid = idx[:, -1] < store.num_ranks
    rows = store.tokens[np.minimum(idx, store.num_ranks - 1)]
    valid &= ~np.any(rows == store.fmt.separator, axis=1)
    return rows, valid


def children(store: MemoryStore, lo: int, hi: int, depth: int):
    """Next tokens of the prefix spanning ranks [lo, hi) at `depth`, with their intervals."""
    keys, los, his = ngram_runs(store, lo, hi, depth, 1)
    return keys[:, 0], los, his
