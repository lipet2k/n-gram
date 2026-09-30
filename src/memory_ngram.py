"""The in-memory prefix cache and the two ways of filling it.

A cache maps a token prefix to its rank interval in the table. A fixed cache holds every
n-gram up to a depth (unigram, bigram, trigram). The greedy cache holds every n-gram up
to a floor depth and then spends the rest of its budget expanding the cached prefix with
the largest interval, the rule from the post. Entries are appended in the order a policy
chooses them, and a greedy build to one budget is a prefix of the build to any larger
budget, so one build serves every smaller budget through `limit_for_bytes`.
"""

from __future__ import annotations

import heapq
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from src.config import IndexFormat
from src.search import children, ngram_runs
from src.store import MemoryStore

NEVER = 1 << 62


@dataclass(frozen=True)
class Hit:
    depth: int
    lo: int
    hi: int
    exact: bool
    expanded: bool

    @property
    def count(self) -> int:
        return self.hi - self.lo


class Cache:
    """Prefix to interval map with byte accounting and a record of when each node was fully expanded."""

    def __init__(self, num_ranks: int, fmt: IndexFormat, bound_bytes: int):
        self.num_ranks = num_ranks
        self.fmt = fmt
        self.bound_bytes = bound_bytes
        self.index: dict[bytes, int] = {}
        self.keys: list[bytes] = []
        self.lo = array("q")
        self.hi = array("q")
        self.expanded_at = array("q")
        self.cum_bytes = array("q")
        self.max_depth = 0
        self.floor_entries = 1
        self.add(b"", 0, num_ranks)

    def __len__(self) -> int:
        return len(self.keys)

    def entry_bytes(self, depth: int) -> int:
        """Compact cost of one entry: the key's tokens plus two bounds. The root is free."""
        return 0 if depth == 0 else self.fmt.token_bytes * depth + self.bound_bytes

    def add(self, key: bytes, lo: int, hi: int) -> int:
        idx = len(self.keys)
        depth = len(key) // self.fmt.token_bytes
        self.index[key] = idx
        self.keys.append(key)
        self.lo.append(lo)
        self.hi.append(hi)
        self.expanded_at.append(NEVER)
        self.cum_bytes.append(
            (self.cum_bytes[-1] if idx else 0) + self.entry_bytes(depth)
        )
        self.max_depth = max(self.max_depth, depth)
        return idx

    def depth(self, idx: int) -> int:
        return len(self.keys[idx]) // self.fmt.token_bytes

    def count(self, idx: int) -> int:
        return self.hi[idx] - self.lo[idx]

    def mark_expanded(self, idx: int) -> None:
        """Record that every child of entry `idx` is cached from this point on."""
        self.expanded_at[idx] = len(self.keys)

    def nbytes(self, limit: int | None = None) -> int:
        return self.cum_bytes[(len(self) if limit is None else limit) - 1]

    def limit_for_bytes(self, max_bytes: int) -> int:
        """Number of leading entries that fit in `max_bytes`."""
        return bisect_right(self.cum_bytes, max_bytes)

    def lookup(self, query: bytes, limit: int | None = None) -> Hit:
        """The longest cached prefix of `query` among the first `limit` entries; the root if none."""
        limit = len(self) if limit is None else limit
        width = self.fmt.token_bytes
        n = len(query) // width
        for depth in range(min(n, self.max_depth), 0, -1):
            idx = self.index.get(query[: depth * width])
            if idx is not None and idx < limit:
                return Hit(
                    depth,
                    self.lo[idx],
                    self.hi[idx],
                    depth == n,
                    self.expanded_at[idx] <= limit,
                )
        return Hit(0, 0, self.num_ranks, False, self.expanded_at[0] <= limit)

    def save(self, path: Path | str) -> None:
        lengths = np.fromiter(
            (len(k) for k in self.keys), dtype=np.int64, count=len(self.keys)
        )
        np.savez(
            path,
            blob=np.frombuffer(b"".join(self.keys), dtype=np.uint8),
            lengths=lengths,
            lo=np.frombuffer(self.lo, dtype=np.int64),
            hi=np.frombuffer(self.hi, dtype=np.int64),
            expanded_at=np.frombuffer(self.expanded_at, dtype=np.int64),
            num_ranks=self.num_ranks,
            floor_entries=self.floor_entries,
            token_bytes=self.fmt.token_bytes,
            bound_bytes=self.bound_bytes,
        )

    @classmethod
    def load(cls, path: Path | str, fmt: IndexFormat) -> Cache:
        data = np.load(path)
        if int(data["token_bytes"]) != fmt.token_bytes:
            raise ValueError(
                f"{path} was built for {int(data['token_bytes'])}-byte tokens"
            )
        cache = cls(int(data["num_ranks"]), fmt, int(data["bound_bytes"]))
        blob = data["blob"].tobytes()
        lengths = data["lengths"]
        ends = np.cumsum(lengths).tolist()
        lo, hi = data["lo"].tolist(), data["hi"].tolist()
        for i in range(1, len(lengths)):
            cache.add(blob[ends[i] - int(lengths[i]) : ends[i]], lo[i], hi[i])
        cache.expanded_at = array("q", data["expanded_at"].tolist())
        cache.floor_entries = int(data["floor_entries"])
        return cache


def build_fixed(store: MemoryStore, depth: int, bound_bytes: int) -> Cache:
    """Every 1-gram up to every `depth`-gram. Parents of a complete level are expanded."""
    cache = Cache(store.num_ranks, store.fmt, bound_bytes)
    for d in range(1, depth + 1):
        keys, los, his = ngram_runs(store, 0, store.num_ranks, 0, d)
        width = d * store.fmt.token_bytes
        blob = keys.tobytes()
        for i, (lo, hi) in enumerate(zip(los.tolist(), his.tolist(), strict=True)):
            cache.add(blob[i * width : (i + 1) * width], lo, hi)
        for idx, key in enumerate(cache.keys):
            if len(key) == width - store.fmt.token_bytes:
                cache.mark_expanded(idx)
        logger.info(
            "level {}: {} entries, {} bytes so far", d, len(keys), cache.nbytes()
        )
    return cache


def build_greedy(
    store: MemoryStore, max_bytes: int, bound_bytes: int, floor_depth: int
) -> Cache:
    """All `floor_depth`-grams, then expand the largest interval, most frequent children first."""
    cache = build_fixed(store, floor_depth, bound_bytes)
    cache.floor_entries = len(cache)
    if cache.nbytes() > max_bytes:
        raise ValueError(
            f"the {floor_depth}-gram floor alone needs {cache.nbytes()} bytes"
        )
    heap = [
        (-cache.count(i), i)
        for i in range(len(cache))
        if cache.expanded_at[i] == NEVER and cache.count(i) >= 2
    ]
    heapq.heapify(heap)
    while heap:
        _, idx = heapq.heappop(heap)
        new, complete = _expand(cache, store, max_bytes, idx)
        if not complete:
            break
        for child in new:
            if cache.count(child) >= 2:
                heapq.heappush(heap, (-cache.count(child), child))
    logger.info(
        "greedy from depth {}: {} entries, {} bytes",
        floor_depth,
        len(cache),
        cache.nbytes(),
    )
    return cache


def _expand(
    cache: Cache, store: MemoryStore, max_bytes: int, idx: int
) -> tuple[list[int], bool]:
    """Add every child of entry `idx` that fits, most frequent first. Returns (new indices, all fit)."""
    toks, los, his = children(store, cache.lo[idx], cache.hi[idx], cache.depth(idx))
    order = np.argsort(los - his, kind="stable")
    toks, los, his = toks[order], los[order], his[order]
    room = (max_bytes - cache.nbytes()) // cache.entry_bytes(cache.depth(idx) + 1)
    n = int(min(len(toks), room))
    key, blob, width = cache.keys[idx], toks.tobytes(), cache.fmt.token_bytes
    new = [
        cache.add(key + blob[width * i : width * (i + 1)], lo, hi)
        for i, (lo, hi) in enumerate(
            zip(los[:n].tolist(), his[:n].tolist(), strict=True)
        )
    ]
    complete = n == len(toks)
    if complete:
        cache.mark_expanded(idx)
    return new, complete
