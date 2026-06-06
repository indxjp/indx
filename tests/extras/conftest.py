"""Harness for the **extras** tier (the ``extras`` marker).

These tests exercise an optional backend's *real* third-party library — the one its
``indx[<extra>]`` install pulls in — with no API key. Some (docling, unstructured, e5,
bge) download public models from Hugging Face on first run, so the tier is opt-in
(nightly/manual) just like ``live`` and is excluded from the default offline suite.

Each test additionally self-skips via :func:`requires_lib` when the library isn't
importable, so the directory stays green even if someone runs ``pytest -m extras`` in an
environment that only has *some* of the extras installed (which is exactly how the e2e
workflow drives it: one extra installed per matrix leg).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def requires_lib() -> Callable[[str], None]:
    """Return a helper that skips the test unless an import name is available.

    Uses :func:`importlib.util.find_spec` (the same check ``require_extra`` performs) so a
    leg whose extra wasn't installed skips cleanly instead of erroring in the vendor SDK::

        def test_x(requires_lib):
            requires_lib("chromadb")
    """

    def _require(module: str) -> None:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"optional library {module!r} is not installed")

    return _require


@pytest.fixture
def text_corpus(tmp_path: Path) -> Path:
    """A tiny two-file plain-text tree (for embedder / store / writer e2e)."""
    handbook = tmp_path / "handbook"
    handbook.mkdir()
    (handbook / "onboarding.md").write_text(
        "# Onboarding\n\nNew hires complete payroll setup and choose benefits in week one.\n"
    )
    engineering = tmp_path / "engineering"
    engineering.mkdir()
    (engineering / "search-api.md").write_text(
        "# Search API\n\nThe service exposes a REST endpoint that ranks documents by relevance.\n"
    )
    return tmp_path


@pytest.fixture
def html_corpus(tmp_path: Path) -> Path:
    """A small HTML file — the one input format markitdown/docling/unstructured all parse."""
    (tmp_path / "page.html").write_text(
        "<html><body>"
        "<h1>Quarterly Note</h1>"
        "<p>Revenue grew across every region.</p>"
        "<p>The engineering team shipped the new search ranking model.</p>"
        "</body></html>\n"
    )
    return tmp_path
