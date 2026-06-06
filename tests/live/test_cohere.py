"""Live e2e against the real Cohere Embed API (``COHERE_API_KEY``).

Cohere is an embedder backend, so a *full pipeline build* (``--embedder cohere``)
exercises it for real: the embed-pack stage calls Cohere's embed endpoint and
``space.search`` re-embeds the query against the same live model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder

pytestmark = pytest.mark.live


def test_cohere_embedder_build_and_query(require_env, live_corpus: Path) -> None:
    require_env("COHERE_API_KEY")

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="cohere", store="jsonl"
    ).run(live_corpus)

    assert space.chunks, "build produced no chunks"
    vectors = [c.embedding for c in space.chunks if c.embedding is not None]
    assert len(vectors) == len(space.chunks), "every chunk should carry a real embedding"
    width = len(vectors[0])
    assert width > 0 and all(len(v) == width for v in vectors), "ragged/empty embeddings"

    hits = space.search("ranking documents by relevance", k=2)
    assert hits, "live retrieval returned no hits"
    assert hits[0].source is not None and hits[0].source.path


def test_cohere_embedder_direct(require_env) -> None:
    require_env("COHERE_API_KEY")

    embedder = get_embedder("cohere")
    vectors = embedder.embed(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    width = len(vectors[0])
    assert width > 0 and len(vectors[1]) == width
    assert all(isinstance(x, float) for x in vectors[0])
