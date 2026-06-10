"""Regression: an invalid --format must fail fast, before the pipeline runs.

cfg.output.format is a free-form string. Previously the output writer slot was resolved only
at the final write step — after walk/parse/enrich/embed had all executed — so a typo in
--format wasted a whole build (and, on cloud stacks, real API spend) before reporting a
trivially-detectable config error. The fix resolves/validates the writer immediately after
load_config; a bad format now raises the same RegistryError (exit code 3) up front.

Runs the offline core stack so no network/model is touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from indx.cli.app import app

runner = CliRunner()

OFFLINE = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nbody\n")
    return src


def test_bad_format_fails_before_pipeline_runs(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    # If validation were deferred, pipeline.run() would execute every stage first. Patch it to
    # blow up loudly so the test fails if the writer is resolved too late.
    with patch("indx.pipeline.DirectoryPipeline.run") as run:
        run.side_effect = AssertionError("pipeline.run must not be called for a bad --format")
        result = runner.invoke(
            app,
            ["build", str(src), "--out", str(out), "--format", "bogusfmt", *OFFLINE],
        )
    # Same error/exit code as before, just reported up front.
    assert result.exit_code == 3
    assert "unknown writer 'bogusfmt'" in result.output
    # No stage work and no output dir from a config error that is knowable before any work.
    assert "stage:" not in result.output
    assert not out.exists()
    run.assert_not_called()


def test_valid_format_still_builds(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--format", "jsonl", *OFFLINE]
    )
    assert result.exit_code == 0
    assert (out / "chunks.jsonl").is_file()
