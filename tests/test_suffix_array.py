from itertools import pairwise

import numpy as np

from src.suffix_array import pointer_size

from .conftest import CFG

FMT = CFG.fmt


def test_pointer_size_follows_infini_gram():
    assert pointer_size(200) == 1
    assert pointer_size(256) == 1
    assert pointer_size(257) == 2
    assert pointer_size(1 << 32) == 4
    assert pointer_size((1 << 32) + 1) == 5


def test_table_is_a_sorted_permutation_of_aligned_offsets(store, ds_bytes):
    ptrs = store.ptrs(0, store.num_ranks)
    assert sorted(ptrs.tolist()) == list(range(0, len(ds_bytes), FMT.token_bytes))
    for a, b in pairwise(ptrs):
        assert ds_bytes[a:] < ds_bytes[b:]
    assert store.ptr_size == pointer_size(len(ds_bytes))
    assert np.array_equal(store.tokens, np.frombuffer(ds_bytes, dtype=FMT.dtype))
