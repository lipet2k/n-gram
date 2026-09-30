import numpy as np

from src.queries import (
    build_query_sets,
    load_query_sets,
    sample_windows,
    save_query_sets,
)

from .conftest import CFG

FMT = CFG.fmt


def test_sample_windows_avoid_the_separator(token_stream):
    rng = np.random.default_rng(0)
    rows = sample_windows(token_stream, 4, 50, rng, FMT.separator)
    assert rows.shape == (50, 4) and rows.dtype == np.dtype(FMT.dtype)
    assert not np.any(rows == FMT.separator)
    stream = token_stream.tolist()
    for row in rows.tolist():
        assert any(stream[s : s + 4] == row for s in range(len(stream) - 3))


def test_query_sets_round_trip(token_stream, tmp_path):
    sets = build_query_sets(token_stream, CFG.queries, FMT.separator)
    assert list(sets) == CFG.queries.lengths
    save_query_sets(tmp_path / "queries.npz", sets)
    loaded = load_query_sets(tmp_path / "queries.npz")
    assert all(np.array_equal(loaded[n], sets[n]) for n in sets)
