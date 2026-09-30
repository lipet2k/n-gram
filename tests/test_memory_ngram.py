import numpy as np
import pytest

from src.experiment import resolve
from src.memory_ngram import NEVER, Cache, build_fixed, build_greedy
from src.search import find, ngram_runs

from .conftest import CFG, VOCAB

FMT = CFG.fmt
BOUND = CFG.caches.bound_bytes


@pytest.fixture(scope="module")
def caches(store):
    fixed = {
        name: build_fixed(store, depth, BOUND)
        for name, depth in (("unigram", 1), ("bigram", 2), ("trigram", 3))
    }
    top = fixed["trigram"].nbytes()
    fixed["greedy2"] = build_greedy(store, top, BOUND, floor_depth=2)
    fixed["greedy0"] = build_greedy(store, top, BOUND, floor_depth=0)
    return fixed


def queries(token_stream, rng, k=40):
    out = []
    for n in (1, 2, 3, 4, 6):
        for s in rng.integers(0, len(token_stream) - n, size=k):
            if FMT.separator not in token_stream[s : s + n]:
                out.append(token_stream[s : s + n])
        out.extend(VOCAB[rng.integers(0, len(VOCAB), size=n)] for _ in range(k // 4))
    return out


def limits(cache):
    """The whole cache plus a few snapshots inside it."""
    top = cache.nbytes()
    return [len(cache)] + [
        cache.limit_for_bytes(b) for b in (top // 7, top // 3, top // 2)
    ]


def test_caches_are_prefix_closed_and_accounted(caches):
    width = FMT.token_bytes
    for name, cache in caches.items():
        assert cache.keys[0] == b""
        assert cache.nbytes() == sum(
            cache.entry_bytes(cache.depth(i)) for i in range(len(cache))
        )
        assert list(cache.cum_bytes) == sorted(cache.cum_bytes)
        for idx in range(1, len(cache)):
            key = cache.keys[idx]
            assert FMT.separator_bytes not in [
                key[i : i + width] for i in range(0, len(key), width)
            ], name
            assert cache.index[key[:-width]] < idx, name


def test_fixed_caches_hold_exactly_every_level(store, caches):
    for name, depth in (("unigram", 1), ("bigram", 2), ("trigram", 3)):
        cache = caches[name]
        for d in range(1, depth + 1):
            keys, los, his = ngram_runs(store, 0, store.num_ranks, 0, d)
            for row, lo, hi in zip(keys, los, his, strict=True):
                idx = cache.index[row.tobytes()]
                assert (cache.lo[idx], cache.hi[idx]) == (lo, hi)
            assert sum(1 for k in cache.keys if len(k) == d * FMT.token_bytes) == len(
                keys
            )
        assert cache.max_depth == depth
        assert all(
            (cache.expanded_at[i] <= len(cache)) == (cache.depth(i) < depth)
            for i in range(len(cache))
        )


def test_cached_search_matches_uncached(store, token_stream, caches):
    rng = np.random.default_rng(11)
    tests = queries(token_stream, rng)
    for name, cache in caches.items():
        for limit in limits(cache):
            cached_total = uncached_total = 0
            for query in tests:
                store.begin_query()
                expected = find(store, FMT.encode(query))
                uncached_total += store.steps
                store.begin_query()
                got = resolve(store, cache, FMT.encode(query), limit)
                cached_total += store.steps
                assert got[1] - got[0] == expected[1] - expected[0], (name, limit)
                if got[1] > got[0]:
                    assert got == expected, (name, limit)
            assert cached_total <= uncached_total, (name, limit)


def test_expanded_flag_means_every_child_is_cached(store, caches):
    for name, cache in caches.items():
        for limit in limits(cache):
            for idx in range(limit):
                if cache.expanded_at[idx] > limit:
                    continue
                for tok in VOCAB:
                    child = cache.keys[idx] + FMT.token(int(tok))
                    lo, hi = find(store, child)
                    if hi > lo:
                        assert cache.index.get(child, NEVER) < limit, (
                            name,
                            limit,
                            cache.keys[idx],
                            tok,
                        )


def test_greedy_floor_holds_every_bigram_within_the_trigram_budget(caches):
    greedy, bigram, trigram = caches["greedy2"], caches["bigram"], caches["trigram"]
    assert greedy.keys[: greedy.floor_entries] == bigram.keys
    assert greedy.nbytes(greedy.floor_entries) == bigram.nbytes()
    assert bigram.nbytes() < greedy.nbytes() <= trigram.nbytes()
    assert caches["greedy0"].floor_entries == 1


@pytest.mark.parametrize("name", ["greedy0", "greedy2"])
def test_greedy_expands_largest_intervals_first(caches, name):
    cache = caches[name]
    floor = 0 if name == "greedy0" else 2
    frontier = max(floor, 1)
    deep = [
        i
        for i in range(len(cache))
        if cache.depth(i) >= frontier and cache.count(i) >= 2
    ]
    expanded = [cache.count(i) for i in deep if cache.expanded_at[i] < NEVER]
    waiting = [cache.count(i) for i in deep if cache.expanded_at[i] == NEVER]
    assert expanded and waiting
    assert min(expanded) >= max(waiting)


@pytest.mark.parametrize("floor", [0, 2])
def test_snapshot_equals_a_direct_build(store, caches, floor):
    full = caches[f"greedy{floor}"]
    for budget in (full.nbytes() // 2, full.nbytes() * 3 // 4):
        small = build_greedy(store, budget, BOUND, floor_depth=floor)
        limit = full.limit_for_bytes(budget)
        assert len(small) == limit and small.keys == full.keys[:limit]
        assert list(small.lo) == list(full.lo[:limit]) and list(small.hi) == list(
            full.hi[:limit]
        )
        assert [small.expanded_at[i] <= len(small) for i in range(limit)] == [
            full.expanded_at[i] <= limit for i in range(limit)
        ]


def test_partial_expansion_keeps_the_most_frequent_children(store, caches):
    bigram = caches["bigram"]
    budget = bigram.nbytes() + 5 * bigram.entry_bytes(3)
    cache = build_greedy(store, budget, BOUND, floor_depth=2)
    added = [i for i in range(len(cache)) if cache.depth(i) == 3]
    assert len(added) == 5 and cache.nbytes() == budget
    parent = cache.index[cache.keys[added[0]][: 2 * FMT.token_bytes]]
    assert cache.count(parent) == max(
        cache.count(i) for i in range(len(cache)) if cache.depth(i) == 2
    )
    _keys, los, his = ngram_runs(store, cache.lo[parent], cache.hi[parent], 2, 1)
    assert sorted(cache.count(i) for i in added) == sorted(
        sorted((his - los).tolist(), reverse=True)[:5]
    )


def test_save_and_load_round_trip(store, token_stream, caches, tmp_path):
    rng = np.random.default_rng(5)
    tests = queries(token_stream, rng, k=10)
    for name, cache in caches.items():
        path = tmp_path / f"{name}.npz"
        cache.save(path)
        loaded = Cache.load(path, FMT)
        assert len(loaded) == len(cache) and loaded.floor_entries == cache.floor_entries
        assert list(loaded.cum_bytes) == list(cache.cum_bytes)
        assert list(loaded.expanded_at) == list(cache.expanded_at)
        for query in tests:
            for limit in limits(cache):
                assert loaded.lookup(FMT.encode(query), limit) == cache.lookup(
                    FMT.encode(query), limit
                )


def test_root_only_cache_searches_everything(store, token_stream):
    cache = Cache(store.num_ranks, FMT, BOUND)
    hit = cache.lookup(FMT.encode(token_stream[:3]))
    assert (hit.depth, hit.lo, hit.hi, hit.exact, hit.expanded) == (
        0,
        0,
        store.num_ranks,
        False,
        False,
    )
