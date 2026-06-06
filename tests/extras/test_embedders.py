"""Extras e2e for the optional local EMBEDDER backends, against their real models.

Both download a public model from Hugging Face on first run (no API key), so they are
``slow``. Each is exercised two ways: a direct ``embed`` round-trip (one vector per text,
consistent positive width) and a full pipeline build that stamps real embeddings onto
every chunk.

* ``e5``  — ``sentence-transformers`` (``intfloat/e5-*``).
* ``bge`` — ``FlagEmbedding`` (``BAAI/bge-m3``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder

pytestmark = [pytest.mark.extras, pytest.mark.slow]


def _assert_real_vectors(vectors: list[list[float]], n: int) -> None:
    assert len(vectors) == n, "expected one vector per input text"
    width = len(vectors[0])
    assert width > 0, "embedding has zero width"
    assert all(len(v) == width for v in vectors), "ragged embedding widths"
    assert all(isinstance(x, float) for x in vectors[0]), "non-float embedding component"


def test_e5_direct_and_build(requires_lib, text_corpus: Path) -> None:
    requires_lib("sentence_transformers")

    _assert_real_vectors(get_embedder("e5").embed(["alpha beta", "gamma delta"]), 2)

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="e5", store="jsonl"
    ).run(text_corpus)
    vecs = [c.embedding for c in space.chunks if c.embedding is not None]
    assert vecs and len(vecs) == len(space.chunks), "every chunk should carry an embedding"


def test_bge_direct_and_build(requires_lib, text_corpus: Path) -> None:
    requires_lib("FlagEmbedding")

    _assert_real_vectors(get_embedder("bge-m3").embed(["alpha beta", "gamma delta"]), 2)

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="bge-m3", store="jsonl"
    ).run(text_corpus)
    vecs = [c.embedding for c in space.chunks if c.embedding is not None]
    assert vecs and len(vecs) == len(space.chunks), "every chunk should carry an embedding"
