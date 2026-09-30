"""Measure every cache on search steps and disk latency, then draw the figures."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from loguru import logger

from src.config import Config
from src.figures import depth_name, draw_all
from src.memory_ngram import Cache, build_fixed, build_greedy
from src.queries import load_query_sets
from src.search import find
from src.store import DiskStore, MemoryStore, Store


def cached(cfg: Config, corpus: str, name: str, build) -> Cache:
    """Build a cache once and keep it next to the index."""
    path = cfg.corpus_dir(corpus) / f"cache_{name}.npz"
    if path.exists() and not cfg.run.rebuild:
        return Cache.load(path, cfg.fmt)
    logger.info("building the {} cache", name)
    cache = build()
    cache.save(path)
    return cache


def resolve(
    store: Store, cache: Cache, query: bytes, limit: int | None = None
) -> tuple[int, int]:
    """Longest cached prefix, then search inside its interval unless the answer is already known."""
    hit = cache.lookup(query, limit)
    if hit.exact:
        return hit.lo, hit.hi
    if hit.expanded:
        return hit.lo, hit.lo
    return find(store, query, hit.lo, hit.hi)


def steps_per_query(
    store: Store, cache: Cache, rows: np.ndarray, limit: int | None = None
) -> np.ndarray:
    steps = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        store.begin_query()
        resolve(store, cache, store.fmt.encode(row), limit)
        steps[i] = store.steps
    return steps


def seconds_per_query(
    store: Store, cache: Cache, rows: np.ndarray, limit: int | None = None
) -> np.ndarray:
    seconds = np.zeros(len(rows))
    for i, row in enumerate(rows):
        query = store.fmt.encode(row)
        start = time.perf_counter()
        resolve(store, cache, query, limit)
        seconds[i] = time.perf_counter() - start
    return seconds


def summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
    }


class Bench:
    """Runs one cache configuration over every query set and the disk latency set."""

    def __init__(self, cfg: Config, corpus: str):
        self.store = MemoryStore(
            cfg.corpus_dir(corpus), cfg.fmt, cfg.build.enumerate_chunk_ranks
        )
        self.queries = load_query_sets(cfg.corpus_dir(corpus) / "queries.npz")
        self.latency_rows = self.queries[cfg.latency.query_length][
            : cfg.latency.queries
        ]
        drop_page_cache(cfg, corpus)
        self.disk = DiskStore(cfg.corpus_dir(corpus), cfg.fmt)

    def measure(self, cache: Cache, limit: int | None = None) -> dict:
        steps = {
            str(n): summary(steps_per_query(self.store, cache, rows, limit))
            for n, rows in self.queries.items()
        }
        self.disk.begin_query()
        seconds = seconds_per_query(self.disk, cache, self.latency_rows, limit)
        return {
            "entries": len(cache) if limit is None else limit,
            "bytes": cache.nbytes(limit),
            "steps": steps,
            "latency_ms": summary(seconds * 1e3),
            "ms_per_step": float(seconds.sum() * 1e3 / max(self.disk.steps, 1)),
        }


def reads_served_from_memory(measured: dict) -> bool:
    """A step is two random reads, so under 5 µs per step only the page cache can have served them."""
    return measured["ms_per_step"] < 5e-3


def drop_page_cache(cfg: Config, corpus: str) -> None:
    """Ask the kernel to forget the index files. Linux honours it without root; macOS needs `sudo purge`."""
    if not hasattr(os, "POSIX_FADV_DONTNEED"):
        logger.warning(
            "cannot drop the page cache on {}; run `sudo purge` first for disk timings",
            sys.platform,
        )
        return
    for name in ("tokenized.0", "table.0"):
        fd = os.open(cfg.corpus_dir(corpus) / name, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def run(cfg: Config, corpus: str) -> dict:
    bench = Bench(cfg, corpus)
    store, bound = bench.store, cfg.caches.bound_bytes
    root = Cache(store.num_ranks, cfg.fmt, bound)
    baseline = bench.measure(root)
    from_memory = reads_served_from_memory(baseline)
    if from_memory:
        logger.warning(
            "a search step took {:.1f} µs, so the page cache served the reads",
            baseline["ms_per_step"] * 1e3,
        )
    results: dict = {
        "corpus": corpus,
        "num_ranks": store.num_ranks,
        "index_bytes": store.ds_size + store.num_ranks * store.ptr_size,
        "query_lengths": cfg.queries.lengths,
        "latency": {
            "query_length": cfg.latency.query_length,
            "queries": len(bench.latency_rows),
            "reads_served_from_memory": from_memory,
        },
        "none": baseline,
        "fixed": [],
        "greedy": [],
    }
    fixed_bytes = []
    for depth in cfg.caches.fixed_depths:
        name = depth_name(depth)
        cache = cached(cfg, corpus, name, lambda d=depth: build_fixed(store, d, bound))
        results["fixed"].append({"name": name, "depth": depth} | bench.measure(cache))
        fixed_bytes.append(cache.nbytes())
        logger.info("{}: {} entries, {} bytes", name, len(cache), cache.nbytes())
    budgets = sorted(set(cfg.corpora[corpus].greedy_budgets_bytes) | set(fixed_bytes))
    for floor in cfg.caches.greedy_floor_depths:
        cache = cached(
            cfg,
            corpus,
            f"greedy{floor}",
            lambda f=floor: build_greedy(store, budgets[-1], bound, f),
        )
        points = []
        for budget in budgets:
            limit = cache.limit_for_bytes(budget)
            if limit < cache.floor_entries:
                continue
            points.append({"budget": budget} | bench.measure(cache, limit))
            logger.info(
                "greedy from depth {} at {} bytes: {} entries", floor, budget, limit
            )
        results["greedy"].append({"floor_depth": floor, "points": points})
    bench.disk.close()

    cfg.results_dir.mkdir(exist_ok=True)
    path = cfg.results_dir / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    draw_all(results, cfg.results_dir)
    logger.info("wrote {} and the figures next to it", path)
    return results
