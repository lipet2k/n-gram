"""Byte-level access to an infini-gram style index.

An index directory holds `tokenized.0` (little-endian tokens, each document prefixed
with the separator) and `table.0` (the suffix array as byte offsets, `ptr_size` bytes
each). A store answers the two reads a search step needs, the pointer at a rank and the
bytes at that pointer, and counts the steps.
"""

from __future__ import annotations

import mmap
import os
import sys
from pathlib import Path

import numpy as np

from src.config import IndexFormat


class Store:
    """Geometry and step counting. Subclasses do the raw reads."""

    def __init__(self, index_dir: Path | str, fmt: IndexFormat):
        self.index_dir = Path(index_dir)
        self.fmt = fmt
        self.ds_size = os.path.getsize(self.index_dir / "tokenized.0")
        table_size = os.path.getsize(self.index_dir / "table.0")
        self.num_ranks = self.ds_size // fmt.token_bytes
        if self.num_ranks == 0 or table_size % self.num_ranks:
            raise ValueError(
                f"table.0 has {table_size} bytes for {self.num_ranks} ranks"
            )
        self.ptr_size = table_size // self.num_ranks
        self.steps = 0

    def begin_query(self) -> None:
        self.steps = 0

    def suffix(self, rank: int, nbytes: int) -> bytes:
        """The first `nbytes` bytes of the suffix at `rank`. One search step, two reads."""
        self.steps += 1
        ptr = self._read_ptr(rank)
        return self._read_ds(ptr, min(ptr + nbytes, self.ds_size))

    def _read_ptr(self, rank: int) -> int:
        raise NotImplementedError

    def _read_ds(self, start: int, end: int) -> bytes:
        raise NotImplementedError


class MemoryStore(Store):
    """Both files memory-mapped, plus the vectorized reads the cache builders need."""

    def __init__(self, index_dir: Path | str, fmt: IndexFormat, chunk_ranks: int):
        super().__init__(index_dir, fmt)
        self.chunk_ranks = chunk_ranks
        self.ds = np.memmap(self.index_dir / "tokenized.0", dtype=np.uint8, mode="r")
        self.tokens = self.ds.view(fmt.dtype)
        table = np.memmap(self.index_dir / "table.0", dtype=np.uint8, mode="r")
        self.table = table.reshape(self.num_ranks, self.ptr_size)

    def _read_ptr(self, rank: int) -> int:
        return int.from_bytes(self.table[rank].tobytes(), "little")

    def _read_ds(self, start: int, end: int) -> bytes:
        return self.ds[start:end].tobytes()

    def ptrs(self, lo: int, hi: int) -> np.ndarray:
        """Byte offsets of ranks [lo, hi) as int64. Not counted as steps."""
        padded = np.zeros((hi - lo, 8), dtype=np.uint8)
        padded[:, : self.ptr_size] = self.table[lo:hi]
        return padded.view("<u8").reshape(-1).astype(np.int64)


class DiskStore(Store):
    """Reads through pread on descriptors that bypass the page cache.

    Every read fetches whole pages, at most two. Linux uses O_DIRECT with a
    page-aligned buffer, macOS uses F_NOCACHE.
    """

    def __init__(self, index_dir: Path | str, fmt: IndexFormat):
        super().__init__(index_dir, fmt)
        self.direct = hasattr(os, "O_DIRECT")
        self.table_fd = self._open(self.index_dir / "table.0")
        self.ds_fd = self._open(self.index_dir / "tokenized.0")
        self.buffer = mmap.mmap(-1, 2 * fmt.page_bytes)

    def _open(self, path: Path) -> int:
        flags = os.O_RDONLY | (os.O_DIRECT if self.direct else 0)
        fd = os.open(path, flags)
        if sys.platform == "darwin":
            import fcntl

            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        return fd

    def read_block(self, fd: int, start: int, nbytes: int) -> bytes:
        """Read [start, start + nbytes) via the aligned pages that contain it."""
        page = self.fmt.page_bytes
        if nbytes > page:
            raise ValueError("reads are limited to one page of payload")
        first = start - start % page
        length = -(-(start + nbytes - first) // page) * page
        offset = start - first
        if self.direct:
            os.preadv(fd, [memoryview(self.buffer)[:length]], first)
            return self.buffer[offset : offset + nbytes]
        return os.pread(fd, length, first)[offset : offset + nbytes]

    def _read_ptr(self, rank: int) -> int:
        block = self.read_block(self.table_fd, rank * self.ptr_size, self.ptr_size)
        return int.from_bytes(block, "little")

    def _read_ds(self, start: int, end: int) -> bytes:
        return self.read_block(self.ds_fd, start, end - start)

    def close(self) -> None:
        os.close(self.table_fd)
        os.close(self.ds_fd)
        self.buffer.close()
