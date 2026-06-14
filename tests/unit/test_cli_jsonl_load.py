"""T4 — `inspect`/`query` load a `--format jsonl` output directory, not just a .indx archive.

`resolve_archive` only finds a sealed `.indx` file; a jsonl output dir (manifest.json +
documents.jsonl + chunks.jsonl + relations.jsonl, no .indx) used to fail with "no .indx
archive found". `load_space` reconstructs the space from those shards so both commands work
on either shape (devex-review-2026-06-06 Issue 4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx.cli._render import load_space
from indx.cli.app import app
from indx.errors import ArchiveError

runner = CliRunner()

# Offline core stack: the cloud defaults need API keys; these flags make the build vendor-free.
OFFLINE = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]


def _build_jsonl_dir(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "notes").mkdir(parents=True)
    (src / "notes" / "alpha.md").write_text("# Alpha\n\nThe onboarding handbook lives here.\n")
    (src / "notes" / "beta.md").write_text("# Beta\n\nSecurity policy details.\n")
    out = tmp_path / "jsonl_out"
    result = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--format", "jsonl", *OFFLINE]
    )
    assert result.exit_code == 0, result.output
    # Sanity: this is the jsonl shape (no .indx in sight).
    assert (out / "manifest.json").is_file()
    assert (out / "chunks.jsonl").is_file()
    assert not list(out.glob("*.indx"))
    return out


def test_load_space_reconstructs_jsonl_dir(tmp_path: Path) -> None:
    out = _build_jsonl_dir(tmp_path)
    space = load_space(out)

    # Manifest (header line uses these) plus every shard came back populated.
    assert space.manifest.schema_version == "1"
    assert space.manifest.indx_version
    assert len(space.documents_) == 2
    assert space.chunks
    assert space.relations
    # Embeddings survive the round-trip, so search works against the rebuilt space.
    assert all(c.embedding is not None for c in space.chunks)
    assert space.search("onboarding handbook", k=2)


def test_inspect_command_on_jsonl_dir(tmp_path: Path) -> None:
    out = _build_jsonl_dir(tmp_path)
    result = runner.invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0, result.output
    assert "documents=2" in result.stdout
    assert "schema=1" in result.stdout  # header line from the reconstructed manifest


def test_query_command_on_jsonl_dir(tmp_path: Path) -> None:
    out = _build_jsonl_dir(tmp_path)
    result = runner.invoke(app, ["query", "onboarding handbook", str(out), "-k", "2"])
    assert result.exit_code == 0, result.output
    assert "query:" in result.stdout


def test_load_space_still_reads_indx_archive(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nonboarding handbook content.\n")
    out = tmp_path / "space.indx"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0

    # The .indx file path and a directory containing it both resolve.
    assert load_space(out).documents_
    assert load_space(out.parent).documents_


def test_load_space_rejects_unknown_path(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError):
        load_space(tmp_path / "does-not-exist")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArchiveError):
        load_space(empty)
