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


def test_build_summary_reports_honest_llm_provenance(tmp_path: Path) -> None:
    # Bug #5: the build-time enricher never invokes an LLM, so requesting --llm openai must NOT
    # be reported as llm=openai. Both the --json summary and the sealed manifest must read
    # llm=none (honest provenance), and they must agree. Runs offline (no model is ever called).
    import json
    import zipfile

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("# Alpha\n\nonboarding notes here.\n")
    out = tmp_path / "space.indx"
    built = runner.invoke(
        app,
        [
            "build",
            str(src),
            "--out",
            str(out),
            "--parser",
            "plaintext",
            "--llm",
            "openai",
            "--embedder",
            "hash",
            "--store",
            "jsonl",
            "--json",
        ],
    )
    assert built.exit_code == 0, built.stdout
    summary = json.loads(built.stdout)
    assert summary["components"]["llm"] == "none"
    assert summary["components"]["vlm"] == "none"
    manifest = json.loads(zipfile.ZipFile(out).read("manifest.json"))
    assert manifest["components"]["llm"] == "none"
    # CLI summary and sealed manifest agree on every shared slot.
    for slot in ("parser", "llm", "vlm", "embedder", "store"):
        assert summary["components"][slot] == manifest["components"][slot]


def test_inspect_no_embed_renders_cleanly(tmp_path: Path) -> None:
    # Bug #15: a space built with --through chunk has no embedding model/dim, so the inspect line
    # used to print the malformed `embedding=/None`. It must render `embedding=none` instead.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("# Alpha\n\nonboarding notes here.\n")
    out = tmp_path / "space.indx"
    offline = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]
    built = runner.invoke(
        app, ["build", str(src), "--out", str(out), *offline, "--through", "chunk"]
    )
    assert built.exit_code == 0, built.stdout

    result = runner.invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "embedding=none" in result.stdout
    assert "embedding=/None" not in result.stdout


def test_inspect_embedded_space_shows_model_and_dim(tmp_path: Path) -> None:
    # Regression guard: a normally-embedded space still prints `embedding=<model>/<dim>`.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("# Alpha\n\nonboarding notes here.\n")
    out = tmp_path / "space.indx"
    offline = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *offline]).exit_code == 0

    result = runner.invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "embedding=hash/" in result.stdout
    assert "embedding=none" not in result.stdout


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
