"""T1 — CLI surface via Typer's CliRunner (no subprocess)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from indx import __version__
from indx.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_build_inspect_query_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "notes").mkdir(parents=True)
    (src / "notes" / "alpha.md").write_text("# Alpha\n\nThe onboarding handbook lives here.\n")
    (src / "notes" / "beta.md").write_text("# Beta\n\nSecurity policy details.\n")
    out = tmp_path / "space.indx"

    # Offline core stack (the cloud defaults need OPENAI_API_KEY).
    offline = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *offline]).exit_code == 0
    assert out.exists()

    inspect = runner.invoke(app, ["inspect", str(out)])
    assert inspect.exit_code == 0
    assert "documents=2" in inspect.stdout

    query = runner.invoke(app, ["query", "onboarding handbook", str(out), "-k", "2"])
    assert query.exit_code == 0
    assert "query:" in query.stdout


def test_missing_extra_surfaces_in_cli(tmp_path: Path) -> None:
    src = tmp_path / "s"
    src.mkdir()
    (src / "a.txt").write_text("hi\n")
    # Use core parser/embedder so qdrant (not the cloud-default docling) is the failing slot.
    args = [
        "build",
        str(src),
        "--out",
        str(tmp_path / "o"),
        "--parser",
        "plaintext",
        "--embedder",
        "hash",
        "--llm",
        "none",
        "--store",
        "qdrant",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1  # clean exit, not an uncaught traceback
    assert "pip install indx[qdrant]" in result.output
