"""Typed access to `configs/config.json`, the one place every setting lives.

Nothing in `src` carries its own defaults. Values flow from the file, through `load`,
into explicit function arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs" / "config.json"


@dataclass(frozen=True)
class IndexFormat:
    """How tokens are laid out on disk: infini-gram's little-endian tokens and separator."""

    token_bytes: int
    separator: int
    page_bytes: int

    @property
    def dtype(self) -> str:
        return f"<u{self.token_bytes}"

    @property
    def separator_bytes(self) -> bytes:
        return self.token(self.separator)

    def token(self, value: int) -> bytes:
        return int(value).to_bytes(self.token_bytes, "little")

    def encode(self, tokens) -> bytes:
        """The byte string of a token sequence, the form the table is sorted by."""
        return np.asarray(tokens, dtype=self.dtype).tobytes()


@dataclass(frozen=True)
class BuildConfig:
    tokenize_batch_docs: int
    enumerate_chunk_ranks: int
    table_write_chunk: int


@dataclass(frozen=True)
class TokenizerConfig:
    repo: str
    path: str | None


@dataclass(frozen=True)
class HeldoutConfig:
    split: str
    shard: int
    max_docs: int


@dataclass(frozen=True)
class DataConfig:
    c4_repo: str
    shard_pattern: str
    shards_per_split: dict[str, int]
    tokenizer: TokenizerConfig
    heldout: HeldoutConfig

    def shard_filename(self, split: str, index: int) -> str:
        total = self.shards_per_split[split]
        return self.shard_pattern.format(split=split, index=index, total=total)


@dataclass(frozen=True)
class CacheConfig:
    bound_bytes: int
    fixed_depths: list[int]
    greedy_floor_depths: list[int]


@dataclass(frozen=True)
class CorpusSpec:
    shards: list[int]
    max_tokens: int
    greedy_budgets_bytes: list[int]


@dataclass(frozen=True)
class QueryConfig:
    lengths: list[int]
    per_length: int
    seed: int


@dataclass(frozen=True)
class LatencyConfig:
    queries: int
    query_length: int


@dataclass(frozen=True)
class RunConfig:
    corpus: str
    rebuild: bool


@dataclass(frozen=True)
class Config:
    data_dir: Path
    heldout_dir: Path
    results_dir: Path
    fmt: IndexFormat
    build: BuildConfig
    data: DataConfig
    caches: CacheConfig
    corpora: dict[str, CorpusSpec]
    queries: QueryConfig
    latency: LatencyConfig
    run: RunConfig

    def corpus_dir(self, name: str) -> Path:
        return self.data_dir / name


def load(path: Path | str = DEFAULT_PATH) -> Config:
    raw = json.loads(Path(path).read_text())
    data = raw["data"]
    return Config(
        data_dir=ROOT / raw["paths"]["data"],
        heldout_dir=ROOT / raw["paths"]["heldout"],
        results_dir=ROOT / raw["paths"]["results"],
        fmt=IndexFormat(**raw["index"]),
        build=BuildConfig(**raw["build"]),
        data=DataConfig(
            c4_repo=data["c4_repo"],
            shard_pattern=data["shard_pattern"],
            shards_per_split=data["shards_per_split"],
            tokenizer=TokenizerConfig(**data["tokenizer"]),
            heldout=HeldoutConfig(**data["heldout"]),
        ),
        caches=CacheConfig(**raw["caches"]),
        corpora={name: CorpusSpec(**spec) for name, spec in raw["corpora"].items()},
        queries=QueryConfig(**raw["queries"]),
        latency=LatencyConfig(**raw["latency"]),
        run=RunConfig(**raw["run"]),
    )
