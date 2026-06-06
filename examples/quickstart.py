#!/usr/bin/env python3
"""Quickstart: build and query a knowledge space with the offline core stack.

This script uses *only* the zero-dependency core components, so it runs on a bare
``pip install -e .`` with no extras (no docling/openai/qdrant/torch). It mirrors the
``--offline`` CLI profile by selecting each slot explicitly:

* parser   = ``plaintext`` — splits text files on blank lines, no network.
* enrich   = off (``enrich=False``) and ``llm``/``vlm`` left unused.
* embedder = ``hash``      — deterministic, model-free vectors.
* store    = ``jsonl``     — brute-force cosine search, self-contained.

Run it with the project interpreter::

    .venv/bin/python examples/quickstart.py

It indexes the bundled ``examples/sample_space`` directory, prints the resulting counts,
and runs a semantic query through the SDK — all in memory, fully offline.
"""

from __future__ import annotations

from pathlib import Path

from indx.pipeline import DirectoryPipeline

# The sample corpus lives next to this script, so the example is self-contained.
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_space"


def main() -> None:
    # Construct the pipeline with the offline core stack. Every slot is named explicitly
    # so we never fall back to an extra-gated cloud default (which would raise
    # MissingExtraError on a bare install). enrich=False keeps the LLM/VLM slots unused.
    pipeline = DirectoryPipeline(
        parser="plaintext",
        embedder="hash",
        store="jsonl",
        enrich=False,
        seed=0,
    )

    # run() with `out` omitted keeps the space in memory and returns a KnowledgeSpace.
    space = pipeline.run(SAMPLE_DIR, name="quickstart")

    # Inspect the build via the SDK. `documents()` is a METHOD; the chunks/relations live
    # on plain fields; `stats` is a property that rolls them up.
    stats = space.stats
    print("Built knowledge space from", SAMPLE_DIR)
    print(f"  documents : {stats.documents}")
    print(f"  chunks    : {stats.chunks}")
    print(f"  relations : {stats.relations}")
    print(f"  embedded  : {stats.embeddings} (dim={stats.embed_dim})")

    print("\nDocuments:")
    for doc in space.documents():
        print(f"  - {doc.path} [{doc.doc_type or 'unknown'}]")

    # Query through the SDK. The embedder is resolved from the manifest (here: 'hash'),
    # a self-contained jsonl store is rebuilt from the space's chunks, and the top hits
    # come back as SearchHit objects.
    query = "How does indx stay deterministic and offline?"
    print(f"\nQuery: {query!r}")
    hits = space.search(query, k=3)
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.chunk.text.split())[:80]
        print(f"  {rank}. score={hit.score:.3f}  {snippet}")


if __name__ == "__main__":
    main()
