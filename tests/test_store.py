import numpy as np

from src.store import DiskStore, MemoryStore

from .conftest import CFG

FMT = CFG.fmt


def test_disk_store_reads_match_memory_store(index_dir, store: MemoryStore):
    disk = DiskStore(index_dir, FMT)
    try:
        rng = np.random.default_rng(0)
        for rank in rng.integers(0, store.num_ranks, size=200):
            for nbytes in (2, 7, 20):
                assert disk.suffix(int(rank), nbytes) == store.suffix(int(rank), nbytes)
        assert disk.suffix(store.num_ranks - 1, 8) == store.suffix(
            store.num_ranks - 1, 8
        )
    finally:
        disk.close()


def test_step_counting(store: MemoryStore):
    store.begin_query()
    store.suffix(store.num_ranks // 3, 6)
    store.suffix(0, 2)
    assert store.steps == 2
    store.begin_query()
    assert store.steps == 0
