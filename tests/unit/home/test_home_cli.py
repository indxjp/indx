"""Feature 4 — default-target resolution + ``indx ask`` / ``indx home`` CLI (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx.cli._render import resolve_target
from indx.cli.app import app
from indx.cli.inspect import inspect_command
from indx.cli.query import query_command
from indx.errors import ArchiveError
from indx.home import home_space_path, open_home

runner = CliRunner()


def _seed_home(tmp_path: Path, body: str = "paid leave goes to the manager") -> None:
    """Append a file to the home space via the CLI ``add`` (no explicit space)."""
    f = tmp_path / "seed.txt"
    f.write_text(f"# Seed\n\n{body}.\n", encoding="utf-8")
    res = runner.invoke(app, ["add", str(f)])
    assert res.exit_code == 0, res.output


# --------------------------------------------------------------------- default-target


def test_query_no_space_targets_home(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_home(tmp_path)
    query_command(None, "leave manager", json_out=True)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list) and payload


def test_add_no_space_appends_to_home(tmp_path: Path) -> None:
    assert open_home().stats.documents == 0
    _seed_home(tmp_path)
    assert open_home().stats.documents == 1  # persisted to home


def test_inspect_no_space_shows_home(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed_home(tmp_path)
    inspect_command(None)
    out = capsys.readouterr().out
    assert "space.indx" in out  # the home archive path appears in the header


def test_explicit_space_wins_over_home(tmp_path: Path) -> None:
    from indx import DirectoryPipeline

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("explicit space content", encoding="utf-8")
    archive = tmp_path / "explicit.indx"
    DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output="jsonl"
    ).run(str(src), str(tmp_path / "o.jsonl")).save(str(archive))

    loaded = resolve_target(archive)
    assert loaded.stats.documents == 1
    # home is untouched.
    assert open_home().stats.documents == 0


def test_resolve_target_missing_explicit_path_archiveerror(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError):
        resolve_target(tmp_path / "nope.indx")


# --------------------------------------------------------------------- ask / home CLI


def test_cli_ask_offline(tmp_path: Path) -> None:
    _seed_home(tmp_path, body="onboarding starts with a welcome session")
    res = runner.invoke(app, ["ask", "how do I onboard?"])
    assert res.exit_code == 0, res.output
    assert "Sources" in res.output


def test_cli_ask_json(tmp_path: Path) -> None:
    _seed_home(tmp_path)
    res = runner.invoke(app, ["ask", "leave policy", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert set(payload) >= {"question", "answer", "hits", "llm"}


def test_cli_home_path() -> None:
    res = runner.invoke(app, ["home", "path"])
    assert res.exit_code == 0, res.output
    from indx.home import home_dir

    assert str(home_dir()) in res.output


def test_cli_home_stats() -> None:
    res = runner.invoke(app, ["home", "stats", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert "documents" in payload and "chunks" in payload


def test_cli_home_reset_confirm_guarded(tmp_path: Path) -> None:
    _seed_home(tmp_path)
    assert home_space_path().is_file()

    declined = runner.invoke(app, ["home", "reset"], input="n\n")
    assert declined.exit_code != 0  # aborted
    assert home_space_path().is_file()  # nothing deleted
    assert open_home().stats.documents == 1

    accepted = runner.invoke(app, ["home", "reset", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert open_home().stats.documents == 0


def test_cli_query_no_space(tmp_path: Path) -> None:
    _seed_home(tmp_path)
    res = runner.invoke(app, ["query", "leave manager"])
    assert res.exit_code == 0, res.output


# --------------------------------------------------------------------- exit codes


def test_ask_explicit_missing_archive_exit2(tmp_path: Path) -> None:
    # Bug #13: `ask` now carries exists=True on its SPACE arg (like `query`), so a missing path is
    # a Click usage error (exit 2) at parse time — symmetric with `query`, not a deep exit 4.
    res = runner.invoke(app, ["ask", "q", str(tmp_path / "nope.indx")])
    assert res.exit_code == 2, res.output


def test_ask_unknown_llm_exit3(tmp_path: Path) -> None:
    _seed_home(tmp_path)
    res = runner.invoke(app, ["ask", "q", "--llm", "__nope__"])
    assert res.exit_code == 3, res.output
