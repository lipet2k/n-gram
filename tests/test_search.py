import numpy as np
import pytest

from src.search import children, find, ngram_runs
from src.store import MemoryStore
from src.suffix_array import brute_force_interval

from .conftest import CFG, VOCAB

FMT = CFG.fmt


def sample_queries(token_stream, rng, n, k=25):
    """k windows from the stream plus k random sequences, most of which are absent."""
    present = [
        token_stream[s : s + n]
        for s in rng.integers(0, len(token_stream) - n, size=4 * k)
        if FMT.separator not in token_stream[s : s + n]
    ][:k]
    absent = [VOCAB[rng.integers(0, len(VOCAB), size=n)] for _ in range(k)]
    return present + absent


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_find_matches_brute_force(store, ds_bytes, token_stream, n):
    rng = np.random.default_rng(n)
    for query in sample_queries(token_stream, rng, n):
        store.begin_query()
        expected = brute_force_interval(ds_bytes, FMT.encode(query), FMT.token_bytes)
        assert find(store, FMT.encode(query)) == expected
        assert store.steps <= 3 * (store.num_ranks.bit_length() + 1)


def test_find_inside_a_prefix_interval_agrees(store, token_stream):
    rng = np.random.default_rng(7)
    for query in sample_queries(token_stream, rng, 4):
        for depth in (1, 2, 3):
            lo, hi = find(store, FMT.encode(query[:depth]))
            assert find(store, FMT.encode(query), lo, hi) == find(
                store, FMT.encode(query)
            )


def brute_force_children(token_stream, prefix):
    """Next tokens after every occurrence of `prefix`, with counts, separator excluded."""
    n = len(prefix)
    counts: dict[int, int] = {}
    for s in range(len(token_stream) - n):
        if np.array_equal(token_stream[s : s + n], prefix):
            nxt = int(token_stream[s + n])
            if nxt != FMT.separator:
                counts[nxt] = counts.get(nxt, 0) + 1
    return counts


@pytest.mark.parametrize("prefix", [[], [1], [256], [3, 4], [0, 0, 0], [999]])
def test_children_matches_brute_force(store, token_stream, prefix):
    lo, hi = find(store, FMT.encode(prefix))
    toks, los, his = children(store, lo, hi, len(prefix))
    expected = brute_force_children(token_stream, np.array(prefix, dtype=FMT.dtype))
    assert {
        int(t): int(h - lo_) for t, lo_, h in zip(toks, los, his, strict=True)
    } == expected
    assert np.all(los[1:] >= his[:-1])
    for tok, child_lo, child_hi in zip(toks, los, his, strict=True):
        assert find(store, FMT.encode(list(prefix) + [int(tok)])) == (
            child_lo,
            child_hi,
        )


def brute_force_kgrams(token_stream, k):
    counts: dict[tuple, int] = {}
    for s in range(len(token_stream) - k + 1):
        row = tuple(int(t) for t in token_stream[s : s + k])
        if FMT.separator not in row:
            counts[row] = counts.get(row, 0) + 1
    return counts


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("chunk", [7, None])
def test_ngram_runs_enumerates_every_kgram(store, token_stream, k, chunk):
    keys, los, his = ngram_runs(store, 0, store.num_ranks, 0, k, chunk=chunk)
    found = {
        tuple(int(t) for t in row): int(h - lo)
        for row, lo, h in zip(keys, los, his, strict=True)
    }
    assert found == brute_force_kgrams(token_stream, k)
    assert np.all(los[1:] >= his[:-1])


def test_memory_store_ptrs_match_scalar_reads(store: MemoryStore):
    ptrs = store.ptrs(0, store.num_ranks)
    for rank in range(0, store.num_ranks, 37):
        assert int(ptrs[rank]) == store._read_ptr(rank)
