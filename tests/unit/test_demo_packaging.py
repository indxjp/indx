"""F2 — the `indx demo` packaging story: corpus is resolvable + the runtime guard.

Two distinct guarantees are tested here:

1. **Resource presence** — `importlib.resources` can resolve the bundled `indx.demo.corpus`
   and the expected `*.md`/`*.txt` files are present across `handbook/`, `engineering/`,
   `people/`. This catches an accidental data-file deletion in the source tree.
2. **Graceful failure** — if the packaged corpus is missing/empty at runtime (e.g. a wheel
   that dropped the non-.py data files), `indx demo` raises a clean typed ``IndxError``
   (``PipelineError``) and exits 1 — NOT a raw ``FileNotFoundError``/``StopIteration``.

NOTE: this suite verifies the *source tree* + the runtime guard only. It deliberately does
NOT build a wheel: hatchling/build are not installed in the offline dev venv. The
authoritative "the corpus is actually inside the built wheel" assertion is CI-only — see
``nox -s docker`` and docs/reference/11-packaging.md.
"""

from __future__ import annotations

import importlib.resources as resources
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx.cli.app import _require_demo_corpus, app
from indx.errors import IndxError, PipelineError

runner = CliRunner()

# The corpus is intentionally cross-linked across these three folders (see
# src/indx/demo/__init__.py); the demo's relations view depends on all three shipping.
_EXPECTED_FILES = {
    "handbook/onboarding.md",
    "handbook/welcome.md",
    "engineering/guide.md",
    "engineering/code-review.md",
    "people/team.txt",
    "people/remote-work.md",
    "people/time-off.md",
}


def test_corpus_resource_is_resolvable() -> None:
    """importlib.resources resolves indx.demo/corpus to a real, non-empty directory."""
    corpus_ref = resources.files("indx.demo") / "corpus"
    with resources.as_file(corpus_ref) as corpus_dir:
        assert corpus_dir.is_dir()
        assert any(corpus_dir.rglob("*")), "packaged demo corpus resolved but is empty"


def test_corpus_contains_expected_files() -> None:
    """Every expected sample file ships across handbook/, engineering/, people/.

    Guards against a silent data-file regression that would still let `indx demo` build but
    drop part of the cross-linked corpus.
    """
    corpus_ref = resources.files("indx.demo") / "corpus"
    with resources.as_file(corpus_ref) as corpus_dir:
        present = {
            p.relative_to(corpus_dir).as_posix() for p in corpus_dir.rglob("*") if p.is_file()
        }
    missing = _EXPECTED_FILES - present
    assert not missing, f"demo corpus is missing expected files: {sorted(missing)}"
    # Only .md/.txt data files (no stray .py) make up the corpus.
    assert all(p.endswith((".md", ".txt")) for p in present), present


def test_guard_raises_typed_error_on_empty_corpus(tmp_path: Path) -> None:
    """The corpus guard raises a typed IndxError (PipelineError) for an empty dir."""
    empty = tmp_path / "corpus"
    empty.mkdir()
    with pytest.raises(PipelineError) as excinfo:
        _require_demo_corpus(empty)
    assert isinstance(excinfo.value, IndxError)
    # Actionable message, not a raw stdlib error.
    assert "demo corpus" in str(excinfo.value)


def test_guard_raises_typed_error_on_missing_corpus(tmp_path: Path) -> None:
    """A nonexistent corpus dir is treated the same as an empty one (typed error)."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(PipelineError):
        _require_demo_corpus(missing)


def test_demo_command_clean_exit_when_corpus_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`indx demo` against an empty corpus exits 1 cleanly — no raw traceback leaks.

    Monkeypatches the corpus path the demo command resolves (via ``resources.as_file``) to
    an empty temp dir, simulating a wheel that dropped the data files. The CLI's error
    boundary must render the typed PipelineError and map it to exit 1, never surface a raw
    FileNotFoundError/StopIteration from the directory walk.
    """
    empty = tmp_path / "corpus"
    empty.mkdir()

    @contextmanager
    def fake_as_file(_ref: object) -> Iterator[Path]:
        yield empty

    # Patch the symbol the demo command actually calls (imported as `resources` in app.py).
    monkeypatch.setattr("indx.cli.app.resources.as_file", fake_as_file)

    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 1, result.output
    # CliRunner mixes stderr into output by default; the clean typed message must appear and
    # no traceback class names leak.
    combined = result.output + (str(result.exception) if result.exception else "")
    assert "FileNotFoundError" not in combined
    assert "StopIteration" not in combined
    assert "error:" in result.output and "demo corpus" in result.output
