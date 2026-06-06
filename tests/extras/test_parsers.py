"""Extras e2e for the optional PARSER backends, against their real libraries.

Each parser is exercised by a *full pipeline build* (``--parser X``) over a small HTML
file — the one input format all three handle — with the rest of the stack pinned to the
offline core (``hash`` embedder, ``jsonl`` store) so no API key is needed. A real parse
must yield at least one document with non-empty recovered text.

* ``markitdown``   — pure-Python conversion; fast.
* ``docling``      — downloads layout models on first run (``slow``).
* ``unstructured`` — partitions documents locally (``slow``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline

pytestmark = pytest.mark.extras


def _build_with_parser(parser: str, corpus: Path) -> None:
    space = DirectoryPipeline(
        seed=0, parser=parser, llm="none", embedder="hash", store="jsonl"
    ).run(corpus)

    assert space.documents(), f"{parser} build produced no documents"
    assert space.chunks, f"{parser} build produced no chunks"
    assert any(c.text.strip() for c in space.chunks), f"{parser} produced only empty text"


def test_markitdown_build(requires_lib, html_corpus: Path) -> None:
    requires_lib("markitdown")
    _build_with_parser("markitdown", html_corpus)


@pytest.mark.slow
def test_docling_build(requires_lib, html_corpus: Path) -> None:
    requires_lib("docling")
    _build_with_parser("docling", html_corpus)


@pytest.mark.slow
def test_unstructured_build(requires_lib, html_corpus: Path) -> None:
    requires_lib("unstructured")
    _build_with_parser("unstructured", html_corpus)
