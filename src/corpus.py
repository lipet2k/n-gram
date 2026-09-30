"""Download C4 shards, tokenize them, and write `tokenized.0` and `offset.0`."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from loguru import logger
from tokenizers import Tokenizer

from src.config import BuildConfig, DataConfig, IndexFormat


def download_shard(data: DataConfig, split: str, index: int) -> Path:
    return Path(
        hf_hub_download(
            data.c4_repo, data.shard_filename(split, index), repo_type="dataset"
        )
    )


def iter_documents(
    data: DataConfig, split: str, shards: Iterable[int]
) -> Iterator[str]:
    """Document texts of the given shards in order, downloading each shard when reached."""
    for index in shards:
        path = download_shard(data, split, index)
        logger.info("reading {}", path.name)
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)["text"]


def load_tokenizer(data: DataConfig, fmt: IndexFormat) -> Tokenizer:
    path = data.tokenizer.path or hf_hub_download(data.tokenizer.repo, "tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    if tokenizer.get_vocab_size() >= fmt.separator:
        raise ValueError("the vocabulary must leave the separator id free")
    return tokenizer


def encode_documents(
    docs: Iterable[str], tokenizer: Tokenizer, batch_docs: int
) -> Iterator[list[int]]:
    """Token ids per document, no special tokens, encoded in batches on all cores."""
    batch: list[str] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == batch_docs:
            yield from (
                e.ids for e in tokenizer.encode_batch(batch, add_special_tokens=False)
            )
            batch = []
    if batch:
        yield from (
            e.ids for e in tokenizer.encode_batch(batch, add_special_tokens=False)
        )


def write_index(
    doc_ids: Iterable[list[int]],
    out_dir: Path | str,
    fmt: IndexFormat,
    max_tokens: int | None = None,
    max_docs: int | None = None,
) -> tuple[int, int]:
    """Write documents as `tokenized.0` and `offset.0`, each document prefixed by the separator.

    Stops before the document that would exceed `max_tokens` (separators included) or
    `max_docs`. Returns (tokens written, documents written).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_tokens = num_docs = 0
    with (
        open(out_dir / "tokenized.0", "wb") as ds,
        open(out_dir / "offset.0", "wb") as offsets,
    ):
        for ids in doc_ids:
            if max_docs is not None and num_docs >= max_docs:
                break
            if max_tokens is not None and num_tokens + 1 + len(ids) > max_tokens:
                break
            offsets.write((num_tokens * fmt.token_bytes).to_bytes(8, "little"))
            ds.write(fmt.separator_bytes)
            ds.write(fmt.encode(ids))
            num_tokens += 1 + len(ids)
            num_docs += 1
            if num_docs % 100_000 == 0:
                logger.info("{} documents, {} tokens", num_docs, num_tokens)
    return num_tokens, num_docs


def build_corpus(
    out_dir: Path | str,
    data: DataConfig,
    fmt: IndexFormat,
    build: BuildConfig,
    tokenizer: Tokenizer,
    shards: list[int],
    split: str = "train",
    max_tokens: int | None = None,
    max_docs: int | None = None,
) -> tuple[int, int]:
    docs = iter_documents(data, split, shards)
    ids = encode_documents(docs, tokenizer, build.tokenize_batch_docs)
    num_tokens, num_docs = write_index(ids, out_dir, fmt, max_tokens, max_docs)
    meta = {
        "split": split,
        "shards": shards,
        "tokens": num_tokens,
        "documents": num_docs,
    }
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        "wrote {} tokens from {} documents to {}", num_tokens, num_docs, out_dir
    )
    return num_tokens, num_docs


def load_tokens(index_dir: Path | str, fmt: IndexFormat) -> np.ndarray:
    """The token stream of an index as a read-only memmap."""
    return np.memmap(Path(index_dir) / "tokenized.0", dtype=fmt.dtype, mode="r")
