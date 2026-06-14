"""Feature 3 — CRUD CLI (``indx add`` / ``indx rm`` / ``indx update``).

Offline core stack only (CliRunner). Asserts exit codes per the contract and CLI⇄SDK parity.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from indx.cli.app import app

runner = CliRunner()


def _build(tmp_path: Path) -> tuple[Path, Path]:
    """Build an offline archive; return (src, out_dir)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "policy.txt").write_text("# Policy\n\npaid leave details.\n")
    (src / "guide.txt").write_text("# Guide\n\nonboarding details.\n")
    out = tmp_path / "out"
    res = runner.invoke(app, ["build", str(src), "--out", str(out), "--offline"])
    assert res.exit_code == 0, res.stdout
    return src, out


def test_cli_add_then_query(tmp_path: Path) -> None:
    src, out = _build(tmp_path)
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n")
    archive = out / "handbook.indx"

    res = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive)])
    assert res.exit_code == 0
    assert "added 1 doc" in res.stdout

    res = runner.invoke(app, ["query", "zylophant", str(archive), "-k", "3"])
    assert res.exit_code == 0
    assert "zylophant" in res.stdout.lower()


def test_cli_rm_then_query(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["rm", "policy.txt", str(archive)])
    assert res.exit_code == 0
    assert "removed 1 doc" in res.stdout

    res = runner.invoke(app, ["query", "leave", str(archive), "-k", "5"])
    assert res.exit_code == 0
    # The removed document's source no longer appears in the hit table.
    assert "policy.txt" not in res.stdout


def test_cli_update(tmp_path: Path) -> None:
    src, out = _build(tmp_path)
    archive = out / "handbook.indx"
    (src / "policy.txt").write_text("# Policy\n\nfrobnicate replaced text.\n")
    res = runner.invoke(app, ["update", str(src / "policy.txt"), str(archive)])
    assert res.exit_code == 0
    assert "updated 1 doc" in res.stdout

    res = runner.invoke(app, ["query", "frobnicate", str(archive), "-k", "3"])
    assert res.exit_code == 0
    assert "frobnicate" in res.stdout.lower()


def test_cli_space_is_output_dir(tmp_path: Path) -> None:
    src, out = _build(tmp_path)
    (src / "extra.txt").write_text("# Extra\n\nnote.\n")
    # Pass the output DIRECTORY (not the .indx) — resolves via resolve_archive.
    res = runner.invoke(app, ["add", str(src / "extra.txt"), str(out)])
    assert res.exit_code == 0
    assert "added 1 doc" in res.stdout


def test_cli_add_missing_path_exit_2(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["add", str(tmp_path / "nope.txt"), str(archive)])
    assert res.exit_code == 2


def test_cli_space_missing_archive_exit_4(tmp_path: Path) -> None:
    src, _ = _build(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    res = runner.invoke(app, ["add", str(src / "policy.txt"), str(empty)])
    assert res.exit_code == 4


def test_cli_rm_unknown_target_exit_0(tmp_path: Path) -> None:
    _, out = _build(tmp_path)
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["rm", "nope.txt", str(archive)])
    assert res.exit_code == 0
    assert "removed 0 doc" in res.stdout


def test_cli_add_json_summary(tmp_path: Path) -> None:
    import json

    src, out = _build(tmp_path)
    (src / "extra.txt").write_text("# Extra\n\nnote.\n")
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive), "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["added"]["docs"] == 1
    assert len(payload["changed"]) == 1
