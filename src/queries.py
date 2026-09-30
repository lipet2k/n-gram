"""Query sets: windows of held-out text, one set per query length."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import QueryConfig


def sample_windows(
    tokens: np.ndarray, n: int, k: int, rng: np.random.Generator, separator: int
) -> np.ndarray:
    """k windows of n tokens from uniformly random positions, none containing the separator."""
    picked: list[np.ndarray] = []
    need = k
    while need > 0:
        starts = rng.integers(0, len(tokens) - n + 1, size=2 * need)
        rows = np.asarray(tokens[starts[:, None] + np.arange(n)])
        rows = rows[~np.any(rows == separator, axis=1)]
        picked.append(rows[:need])
        need -= len(rows[:need])
    return np.concatenate(picked)


def build_query_sets(
    heldout: np.ndarray, queries: QueryConfig, separator: int
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(queries.seed)
    return {
        n: sample_windows(heldout, n, queries.per_length, rng, separator)
        for n in queries.lengths
    }


def save_query_sets(path: Path | str, sets: dict[int, np.ndarray]) -> None:
    np.savez(path, **{str(n): rows for n, rows in sets.items()})


def load_query_sets(path: Path | str) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {int(name): data[name] for name in data.files}
