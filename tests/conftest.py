"""Shared fixtures for the indx test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.core.context import SpaceContext
from indx.core.knowledge_space import KnowledgeSpace

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = Path(__file__).resolve().parent / "corpus"


@pytest.fixture(autouse=True)
def _plain_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force plain, color-free, wide Typer/Rich ``--help`` so substring assertions are stable.

    Several tests assert a flag (``--aws``, ``--host``, ``--strict``, …) appears in captured
    ``--help`` text. Typer renders help through Rich, and CI (GitHub Actions) makes Rich
    *force color* — which interleaves ANSI escape codes into the flag text, so ``"--aws" in
    stdout`` is False even though the flag is shown. Rich also sizes its option table to the
    terminal width and collapses cells in a narrow/TTY-less runner. So pin the environment to
    deterministic plain output: drop ``FORCE_COLOR``, set ``NO_COLOR``/``TERM=dumb`` to kill
    ANSI styling, and widen ``COLUMNS`` so nothing wraps. Autouse + monkeypatch so it applies
    everywhere and is restored after each test.
    """
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def tmp_tree(tmp_path: Path) -> Path:
    """A tiny on-disk directory tree for walk/parse tests."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text("# One\n\nAlpha beta gamma.\n")
    (tmp_path / "a" / "two.md").write_text("# Two\n\nDelta epsilon.\n")
    (tmp_path / "root.txt").write_text("Top level note.\n")
    return tmp_path


@pytest.fixture
def empty_context(tmp_path: Path) -> SpaceContext:
    return SpaceContext(root=tmp_path, space=KnowledgeSpace())
