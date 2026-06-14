"""README CLI-prose contract tests (offline, no network/keys).

The shipped CLI is **value-first**: the required positional is the text/path and the
optional SPACE is the *second* positional (verified via ``--help``: ``query [OPTIONS]
TEXT [SPACE]``, ``ask ... QUESTION [SPACE]``, ``add ... PATH [SPACE]``, etc.). These
tests pin the README examples to that real arg order, the real ``compose`` flow (which
edits an *existing* parent manifest), the local-only build-time enrich contract, and the
current package version — so the docs can't silently drift from the code again.
"""

from __future__ import annotations

import re
from pathlib import Path

import indx

README = Path(__file__).resolve().parent.parent / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_query_ask_are_value_first() -> None:
    """No README ``query``/``ask`` example puts the ``.indx`` SPACE before the TEXT (#2)."""
    text = _readme()
    # Anti-pattern: ``indx query ./foo.indx "..."`` (space-first) — the .indx token sits
    # before the quoted text. Value-first is ``indx query "..." ./foo.indx``.
    space_first = re.compile(r"indx\s+(query|ask)\s+\S*\.indx\s+[\"']")
    bad = [ln for ln in text.splitlines() if space_first.search(ln)]
    assert not bad, f"space-first query/ask examples found: {bad}"

    # ...and the value-first form is actually present for both verbs.
    assert re.search(r"indx\s+query\s+[\"'][^\"']+[\"']\s+\./\S*\.indx", text)
    assert re.search(r"indx\s+ask\s+[\"'][^\"']+[\"']\s+\./\S*\.indx", text)


def test_crud_examples_are_value_first() -> None:
    """``add``/``rm``/``update`` placeholders are PATH-first, not ``<space>``-first (#3)."""
    text = _readme()
    space_first = re.compile(r"indx\s+(add|rm|update)\s+<space>")
    bad = [ln for ln in text.splitlines() if space_first.search(ln)]
    assert not bad, f"space-first CRUD placeholders found: {bad}"

    # The corrected value-first placeholders are present.
    assert "indx add <path> <space>" in text
    assert "indx rm <doc|path> <space>" in text
    assert "indx update <path> <space>" in text


def test_compose_example_creates_parent_first() -> None:
    """The ``indx of indx`` block builds the parent before composing (Doc-for-#8).

    ``compose``'s PARENT is ``exists=True`` — it edits an existing manifest and never
    creates one — so the README must show the parent-creation step first.
    """
    text = _readme()
    lines = text.splitlines()
    compose_idx = next(i for i, ln in enumerate(lines) if "indx compose ./all.indx --add" in ln)
    # Some preceding line in the same block must create ./all.indx via --out.
    preceding = "\n".join(lines[max(0, compose_idx - 4) : compose_idx])
    assert re.search(r"--out\s+\./all\.indx", preceding), (
        "compose example must be preceded by a parent-creation `--out ./all.indx` step"
    )


def test_no_build_time_llm_enrichment_claim() -> None:
    """README must not claim build-time LLM enrichment; ``ask`` is synthesis (Doc-for-#5)."""
    text = _readme()
    assert "LLM/VLM enrichment is opt-in" not in text
    # The offline-core note explains synthesis happens at query time via ``ask``.
    assert "ask" in text


def test_status_version_matches_package() -> None:
    """README Status line tracks ``indx.__version__``; no stale 0.0.1 (badge-status)."""
    text = _readme()
    assert f"Alpha (`{indx.__version__}`)" in text
    assert "Alpha (`0.0.1`)" not in text
    # The demo transcript schema/version line is not stale either.
    assert "schema=1 indx=0.0.1" not in text
