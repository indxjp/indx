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


def test_cli_rm_unknown_target_warns_and_exits_1(tmp_path: Path) -> None:
    # Bug #14: a 0-match rm is almost always a typo'd id/path, never a legitimate no-op the way
    # re-adding an unchanged file is — so it warns (not a green ✓) and exits non-zero.
    _, out = _build(tmp_path)
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["rm", "nope.txt", str(archive)])
    assert res.exit_code == 1
    assert "no documents matched" in res.stdout
    assert "removed 0 doc" not in res.stdout


def test_cli_rm_no_match_json_stays_exit_0(tmp_path: Path) -> None:
    # The --json branch is for scripted consumers: it keeps a stable schema and exit 0 even on a
    # 0-match, so the warning/non-zero signal is the human-output path only.
    import json

    _, out = _build(tmp_path)
    archive = out / "handbook.indx"
    res = runner.invoke(app, ["rm", "nope.txt", str(archive), "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["removed"]["ids"] == []
    assert payload["removed"]["docs"] == 0


def test_cli_add_noop_warns_exit_0(tmp_path: Path) -> None:
    # Bug #14: an add that ingests nothing (e.g. a path with no ingestable files) used to print a
    # green ✓ 0-doc line that masked typos — emit a distinct warning while keeping exit 0.
    src, out = _build(tmp_path)
    archive = out / "handbook.indx"
    empty_dir = src / "emptydir"
    empty_dir.mkdir()
    res = runner.invoke(app, ["add", str(empty_dir), str(archive)])
    assert res.exit_code == 0
    assert "no documents added" in res.stdout
    assert "✓" not in res.stdout


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
    # H1: the JSON payload always carries an integer parse_failures count.
    assert payload["parse_failures"] == 0


def test_cli_readd_zero_delta_warns_not_green(tmp_path: Path) -> None:
    # L3: re-adding an already-indexed file can return a non-empty `changed` list while
    # contributing zero real deltas (+0 docs / +0 chunks). That must read as the yellow
    # "no documents added" warning, NOT a contradictory green "added N doc(s) (+0/+0)".
    src, out = _build(tmp_path)
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n")
    archive = out / "handbook.indx"

    first = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive)])
    assert first.exit_code == 0
    assert "added 1 doc" in first.stdout

    # Second add of the identical, already-indexed file: no real deltas.
    second = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive)])
    assert second.exit_code == 0
    assert "no documents added" in second.stdout
    assert "✓" not in second.stdout
    assert "+0 docs" not in second.stdout


def test_cli_add_parse_failure_warns_exit_0(tmp_path: Path) -> None:
    # H1: when a newly-added file reaches the parse stage but yields 0 chunks, add prints a
    # prominent yellow warning naming the count and keeps exit 0 (non-strict).
    from indx.cli import crud as crud_module

    src, out = _build(tmp_path)
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n")
    archive = out / "handbook.indx"

    real_add = crud_module.KnowledgeSpace.add

    def add_with_failure(self: object, arg: str) -> object:
        changed = real_add(self, arg)
        # Simulate a parse failure surfaced by the ingest pipeline (Agent B contract).
        self.parse_failures_ = 1  # type: ignore[attr-defined]
        return changed

    monkey = crud_module.KnowledgeSpace.add
    try:
        crud_module.KnowledgeSpace.add = add_with_failure  # type: ignore[method-assign]
        res = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive)])
    finally:
        crud_module.KnowledgeSpace.add = monkey  # type: ignore[method-assign]
    assert res.exit_code == 0
    assert "1 file(s) could not be parsed" in res.stdout
