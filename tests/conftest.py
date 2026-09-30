"""A tiny index with a small, repetitive vocabulary, including tokens above 255.

Tokens 256 and 257 sort before token 1 in little-endian byte order, which is the order the
table uses, so the tests exercise the byte-order subtlety and not just numeric order.
The index format and build knobs come from the real config file, which also checks it parses.
"""

import numpy as np
import pytest

from src import config
from src.corpus import write_index
from src.store import MemoryStore
from src.suffix_array import build_table

VOCAB = np.array(list(range(10)) + [256, 257, 512], dtype=np.int64)
CFG = config.load()


@pytest.fixture(scope="session")
def cfg():
    return CFG


@pytest.fixture(scope="session")
def fmt():
    return CFG.fmt


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    out = tmp_path_factory.mktemp("index")
    docs = [
        VOCAB[rng.integers(0, len(VOCAB), size=int(rng.integers(3, 50)))].tolist()
        for _ in range(120)
    ]
    write_index(docs, out, CFG.fmt)
    build_table(out, CFG.fmt, CFG.build.table_write_chunk)
    return out


@pytest.fixture(scope="session")
def store(index_dir):
    return MemoryStore(index_dir, CFG.fmt, CFG.build.enumerate_chunk_ranks)


@pytest.fixture(scope="session")
def ds_bytes(index_dir):
    return (index_dir / "tokenized.0").read_bytes()


@pytest.fixture(scope="session")
def token_stream(store):
    return np.asarray(store.tokens)
