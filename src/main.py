"""Run what `configs/config.json` says: build the run corpus if needed, then compare the caches."""

from __future__ import annotations

from loguru import logger

from src import config, corpus, experiment, queries, suffix_array
from src.config import Config


def build_index(cfg: Config, name: str) -> None:
    """Corpus, held-out text, table and query sets for `name`, skipping outputs that exist."""
    spec = cfg.corpora[name]
    out = cfg.corpus_dir(name)
    rebuild = cfg.run.rebuild
    tokenizer = None
    if rebuild or not (out / "tokenized.0").exists():
        tokenizer = corpus.load_tokenizer(cfg.data, cfg.fmt)
        corpus.build_corpus(
            out,
            cfg.data,
            cfg.fmt,
            cfg.build,
            tokenizer,
            spec.shards,
            max_tokens=spec.max_tokens,
        )
    if rebuild or not (cfg.heldout_dir / "tokenized.0").exists():
        tokenizer = tokenizer or corpus.load_tokenizer(cfg.data, cfg.fmt)
        held = cfg.data.heldout
        corpus.build_corpus(
            cfg.heldout_dir,
            cfg.data,
            cfg.fmt,
            cfg.build,
            tokenizer,
            [held.shard],
            held.split,
            max_docs=held.max_docs,
        )
    if rebuild or not (out / "table.0").exists():
        suffix_array.build_table(out, cfg.fmt, cfg.build.table_write_chunk)
    if rebuild or not (out / "queries.npz").exists():
        heldout = corpus.load_tokens(cfg.heldout_dir, cfg.fmt)
        sets = queries.build_query_sets(heldout, cfg.queries, cfg.fmt.separator)
        queries.save_query_sets(out / "queries.npz", sets)
        logger.info("wrote {} query sets to {}", len(sets), out / "queries.npz")


def main() -> None:
    cfg = config.load()
    build_index(cfg, cfg.run.corpus)
    experiment.run(cfg, cfg.run.corpus)


if __name__ == "__main__":
    main()
