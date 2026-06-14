"""Feature 2 — CLI compose + federated inspect/query, with the documented exit codes."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from indx import DirectoryPipeline, KnowledgeSpace
from indx.archive import read_archive, write_archive
from indx.cli.app import app

runner = CliRunner()


def _offline() -> DirectoryPipeline:
    return DirectoryPipeline(parser="plaintext", llm="none", embedder="hash", store="jsonl")


def _build(root: Path, files: list[tuple[str, str]], dest: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files:
        (root / name).write_text(body, encoding="utf-8")
    write_archive(_offline().run(str(root)), dest)


def _scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    p = tmp_path / "p.indx"
    _build(tmp_path / "ca", [("a.txt", "alpha onboarding apple")], a)
    _build(tmp_path / "cb", [("b.txt", "beta payroll banana")], b)
    _build(tmp_path / "cp", [("p.txt", "parent root memo")], p)
    return p, a, b


def test_compose_adds_children(tmp_path: Path) -> None:
    p, a, b = _scene(tmp_path)
    a_bytes, b_bytes = a.read_bytes(), b.read_bytes()
    result = runner.invoke(app, ["compose", str(p), "--add", str(a), "--add", str(b)])
    assert result.exit_code == 0, result.output
    assert "added" in result.output
    reloaded = read_archive(p)
    assert [c.name for c in reloaded.manifest.children] == ["a", "b"]
    # refs stored relative to the parent dir; child files untouched.
    assert reloaded.manifest.children[0].ref == "a.indx"
    assert a.read_bytes() == a_bytes
    assert b.read_bytes() == b_bytes


def test_compose_no_args_is_usage_error(tmp_path: Path) -> None:
    p, _a, _b = _scene(tmp_path)
    result = runner.invoke(app, ["compose", str(p)])
    assert result.exit_code == 2


def test_compose_missing_child_is_archive_error(tmp_path: Path) -> None:
    p, _a, _b = _scene(tmp_path)
    # --add with exists=True at the Typer layer rejects a missing path as exit 2; to exercise the
    # ArchiveError path, point at a non-archive existing file.
    bogus = tmp_path / "bogus.indx"
    bogus.write_text("not a zip")
    result = runner.invoke(app, ["compose", str(p), "--add", str(bogus)])
    assert result.exit_code == 4, result.output


def test_compose_duplicate_name_is_usage_error(tmp_path: Path) -> None:
    p, a, b = _scene(tmp_path)
    result = runner.invoke(
        app, ["compose", str(p), "--add", str(a), "--name", "dup", "--add", str(b), "--name", "dup"]
    )
    assert result.exit_code == 2


def test_compose_remove(tmp_path: Path) -> None:
    p, a, _b = _scene(tmp_path)
    runner.invoke(app, ["compose", str(p), "--add", str(a), "--name", "eng"])
    result = runner.invoke(app, ["compose", str(p), "--remove", "eng"])
    assert result.exit_code == 0, result.output
    assert read_archive(p).manifest.children == []


def test_compose_no_pin(tmp_path: Path) -> None:
    p, a, _b = _scene(tmp_path)
    runner.invoke(app, ["compose", str(p), "--add", str(a), "--no-pin"])
    assert read_archive(p).manifest.children[0].sha256 is None


def _composed(tmp_path: Path) -> Path:
    p, a, b = _scene(tmp_path)
    runner.invoke(
        app, ["compose", str(p), "--add", str(a), "--name", "eng", "--add", str(b), "--name", "des"]
    )
    return p


def test_inspect_federates_and_shows_tree(tmp_path: Path) -> None:
    p = _composed(tmp_path)
    parent_only_docs = len(read_archive(p).documents_)
    result = runner.invoke(app, ["inspect", str(p)])
    assert result.exit_code == 0, result.output
    assert "Children" in result.output
    assert "eng" in result.output and "des" in result.output
    # federated doc count > parent-only
    fed = KnowledgeSpace.load(str(p)).stats.documents
    assert fed > parent_only_docs


def test_inspect_no_children(tmp_path: Path) -> None:
    p = _composed(tmp_path)
    result = runner.invoke(app, ["inspect", str(p), "--no-children"])
    assert result.exit_code == 0, result.output
    assert "Children" not in result.output


def test_query_federates(tmp_path: Path) -> None:
    p = _composed(tmp_path)
    result = runner.invoke(app, ["query", "payroll", str(p)])
    assert result.exit_code == 0, result.output
    assert "payroll" in result.output


def test_query_no_children_excludes(tmp_path: Path) -> None:
    p = _composed(tmp_path)
    # Parent-only query: the child-only "payroll/banana" body must not appear in any hit row.
    # (The query echo in the table title always contains the query text, so assert on body words.)
    result = runner.invoke(app, ["query", "payroll", str(p), "--no-children"])
    assert result.exit_code == 0, result.output
    assert "banana" not in result.output
    assert "p.txt" in result.output


def test_inspect_missing_child_renders_state_query_fails(tmp_path: Path) -> None:
    p, a, _b = _scene(tmp_path)
    runner.invoke(app, ["compose", str(p), "--add", str(a), "--name", "eng", "--no-pin"])
    a.unlink()  # ship parent without the child
    insp = runner.invoke(app, ["inspect", str(p)])
    assert insp.exit_code == 0, insp.output
    assert "missing" in insp.output
    # but a federated query MUST read the child and therefore fail.
    q = runner.invoke(app, ["query", "anything", str(p)])
    assert q.exit_code == 4, q.output
