"""CLI wiring for the Feature 5 build filter flags (--include/--exclude/--ext/--max-*/--name-glob).

Exercises the Typer command end-to-end on the offline core stack: repeatable flags, CLI-over-config
precedence, config-only application, the dry-run skip count, and the typed exit codes for bad size /
bad int. No network or heavy extras.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from indx.cli.app import app

runner = CliRunner()

OFFLINE = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("alpha\n")
    (src / "b.txt").write_text("beta\n")
    (src / "c.rst").write_text("gamma\n")
    return src


def _docs(out: Path) -> list[str]:
    from indx import KnowledgeSpace

    space = KnowledgeSpace.load(str(out / "handbook.indx"))
    return sorted(d.path for d in space.documents())


def test_repeatable_ext_flags(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--ext", "md", "--ext", "txt", *OFFLINE],
    )
    assert result.exit_code == 0, result.output
    assert _docs(out) == ["a.md", "b.txt"]


def test_cli_ext_overrides_config(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    cfg = tmp_path / "indx.toml"
    cfg.write_text('[walk]\next = ["rst"]\n')
    # config says rst, CLI says md → CLI wins
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--config", str(cfg), "--ext", "md", *OFFLINE],
    )
    assert result.exit_code == 0, result.output
    assert _docs(out) == ["a.md"]


def test_config_only_filter_applies(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    cfg = tmp_path / "indx.toml"
    cfg.write_text('[walk]\next = ["md"]\n')
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--config", str(cfg), *OFFLINE],
    )
    assert result.exit_code == 0, result.output
    assert _docs(out) == ["a.md"]


def test_dry_run_reflects_filter(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--dry-run", "--ext", "md", *OFFLINE],
    )
    assert result.exit_code == 0, result.output
    assert "skipped by filter" in result.stdout
    assert "a.md" in result.stdout
    assert not out.exists()  # dry run writes nothing


def test_bad_size_is_config_error_exit_3(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--max-size", "2gigs", *OFFLINE],
    )
    assert result.exit_code == 3
    assert not out.exists()


def test_bad_max_files_is_usage_error_exit_2(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--max-files", "notanint", *OFFLINE],
    )
    assert result.exit_code == 2


def test_negative_max_files_is_config_error_exit_3(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--max-files", "-1", *OFFLINE],
    )
    assert result.exit_code == 3
    assert not out.exists()


def test_unknown_walk_key_is_config_error_exit_3(tmp_path: Path) -> None:
    src = _src(tmp_path)
    out = tmp_path / "out"
    cfg = tmp_path / "indx.toml"
    cfg.write_text('[walk]\nexclud = ["x"]\n')
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--config", str(cfg), *OFFLINE],
    )
    assert result.exit_code == 3
