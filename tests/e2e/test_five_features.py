"""Offline end-to-end walkthrough exercising ALL FIVE 2026-06-14 features together.

One module that drives the real public SDK (``indx`` package) *and* the real Typer CLI
(``typer.testing.CliRunner`` against ``indx.cli.app.app``) over the zero-dependency offline
core stack (``plaintext`` parser / ``none`` llm+vlm / ``hash`` embedder / ``jsonl`` store /
``.indx`` writer). No network, no heavy extras, no API key. Every home-mode test isolates
``$INDX_HOME`` to a tmp dir so the real ``~/.indx`` is never touched.

Feature coverage (each test interleaves multiple features so the seams are exercised together):

* **F5** — conditional/filtered directory import: ``--ext``/``--exclude``/``--max-files`` select
  which files enter the space; only the selected files are indexed (the rest are countable skips).
* **F1** — granular per-stage load: ``--through chunk`` persists a chunk-ful, embedding-less space;
  ``KnowledgeSpace.load(members=...)`` / ``load_part`` reads back a single member.
* **F2** — indx-of-indx: ``indx compose`` references two independently-built children; a federated
  ``query`` returns hits from both, and ``stats`` federates over the flattened tree.
* **F3** — CRUD: ``indx add`` / ``indx update`` / ``indx rm`` mutate a sealed archive in place and
  the re-query reflects each change; relations + vectors stay consistent.
* **F4** — home DB: ``$INDX_HOME`` add → query → ask with no explicit space, plus
  ``indx home stats`` / ``indx home reset``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx import Answer, DirectoryPipeline, KnowledgeSpace
from indx.archive import read_archive
from indx.cli.app import app

runner = CliRunner()

# The zero-dependency offline core stack (no extras, no API key).
OFFLINE = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point ``$INDX_HOME`` at a per-test tmp dir; assert the real ~/.indx stays untouched.

    Every test in this module (not only the F4 ones) runs with an isolated home so even an
    accidental default-target resolution can never read or write the user's real home space.
    """
    real_home_archive = Path("~/.indx/space.indx").expanduser()
    existed = real_home_archive.exists()
    monkeypatch.setenv("INDX_HOME", str(tmp_path / "home"))
    yield
    assert real_home_archive.exists() == existed


def _mixed_corpus(root: Path) -> Path:
    """A directory with mixed extensions, a subtree, and a large file — material for the filter."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "alpha.md").write_text(
        "# Alpha\n\nThe onboarding handbook lives here.\n", encoding="utf-8"
    )
    (root / "docs" / "beta.md").write_text(
        "# Beta\n\nSecurity policy and paid leave details.\n", encoding="utf-8"
    )
    (root / "notes.txt").write_text("plain note about payroll\n", encoding="utf-8")
    (root / "data.json").write_text('{"k": "v"}\n', encoding="utf-8")
    # A large .md that --max-size would drop; an excluded subtree under logs/.
    (root / "docs" / "huge.md").write_text("# Huge\n\n" + ("padding " * 5000), encoding="utf-8")
    (root / "logs" / "trace.md").write_text(
        "# Trace\n\nexcluded subtree noise.\n", encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------- F5: filtered import


def test_f5_ext_and_exclude_filter_select_only_wanted_files(tmp_path: Path) -> None:
    """``--ext md --exclude 'logs/*'`` indexes only the non-excluded markdown files (F5)."""
    src = _mixed_corpus(tmp_path / "src")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "build",
            str(src),
            "--out",
            str(out),
            "--ext",
            "md",
            "--exclude",
            "logs/*",
            *OFFLINE,
        ],
    )
    assert result.exit_code == 0, result.output

    space = read_archive(out / "handbook.indx")
    paths = sorted(d.path for d in space.documents_)
    # Only markdown under docs/ — .txt/.json dropped by --ext, logs/ dropped by --exclude.
    assert paths == ["docs/alpha.md", "docs/beta.md", "docs/huge.md"]
    assert all(p.endswith(".md") for p in paths)
    assert not any(p.startswith("logs/") for p in paths)


def test_f5_max_files_caps_deterministically(tmp_path: Path) -> None:
    """``--max-files 2`` keeps the first two files in sorted walk order, deterministically (F5)."""
    src = _mixed_corpus(tmp_path / "src")
    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    args = ["--ext", "md", "--exclude", "logs/*", "--max-files", "2"]
    r1 = runner.invoke(app, ["build", str(src), "--out", str(out1), *args, *OFFLINE])
    r2 = runner.invoke(app, ["build", str(src), "--out", str(out2), *args, *OFFLINE])
    assert r1.exit_code == 0 and r2.exit_code == 0, (r1.output, r2.output)

    s1 = read_archive(out1 / "handbook.indx")
    assert len(s1.documents_) == 2
    # Deterministic selection AND byte-identical archive across runs (NFR-DET-1).
    assert (out1 / "handbook.indx").read_bytes() == (out2 / "handbook.indx").read_bytes()


def test_f5_dry_run_plan_reflects_the_filter(tmp_path: Path) -> None:
    """A ``--dry-run`` build previews exactly the filtered selection and writes nothing (F5)."""
    src = _mixed_corpus(tmp_path / "src")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--ext", "txt", "--dry-run", *OFFLINE],
    )
    assert result.exit_code == 0, result.output
    assert "notes.txt" in result.output
    # Markdown is filtered out of the plan.
    assert "alpha.md" not in result.output
    assert not out.exists()  # dry run produces no archive


# ----------------------------------------------------- F1 + F5: granular load on a filtered build


def test_f1_through_chunk_build_then_selective_load(tmp_path: Path) -> None:
    """``--through chunk`` over a filtered corpus → chunks, no embeddings; load one member (F1)."""
    src = _mixed_corpus(tmp_path / "src")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "build",
            str(src),
            "--out",
            str(out),
            "--ext",
            "md",
            "--exclude",
            "logs/*",
            "--through",
            "chunk",
            *OFFLINE,
        ],
    )
    assert result.exit_code == 0, result.output

    archive = out / "handbook.indx"

    # CLI inspect --json confirms a chunk-ful, embedding-less, relation-less partial space.
    inspected = runner.invoke(app, ["inspect", str(archive), "--json"])
    assert inspected.exit_code == 0, inspected.output
    stats = json.loads(inspected.stdout)
    assert stats["chunks"] > 0
    assert stats["embeddings"] == 0
    assert stats["relations"] == 0

    # SDK selective load: only the requested member is materialized.
    docs_only = KnowledgeSpace.load(str(archive), members=("documents",))
    assert len(docs_only.documents_) > 0
    assert docs_only.chunks == []
    assert docs_only.relations == []

    chunks = KnowledgeSpace.load_part(str(archive), "chunks")
    from indx.core.chunk import Chunk

    assert chunks and all(isinstance(c, Chunk) for c in chunks)
    # The full load matches the chunk count reported by inspect.
    assert len(chunks) == stats["chunks"]


# ----------------------------------------------------------- F2: compose two built archives


def _build_archive(src_dir: Path, files: list[tuple[str, str]], dest: Path) -> None:
    """Build an offline ``.indx`` at ``dest`` from ``files`` written under ``src_dir``."""
    src_dir.mkdir(parents=True, exist_ok=True)
    for name, body in files:
        (src_dir / name).write_text(body, encoding="utf-8")
    pipe = DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output="indx"
    )
    pipe.run(str(src_dir), str(dest))


def test_f2_compose_two_archives_and_query_composite(tmp_path: Path) -> None:
    """Compose two independently-built children under a parent; query federates over both (F2)."""
    eng = tmp_path / "eng.indx"
    des = tmp_path / "des.indx"
    parent = tmp_path / "parent.indx"
    _build_archive(tmp_path / "ce", [("e.txt", "engineering deploys the apple service")], eng)
    _build_archive(tmp_path / "cd", [("d.txt", "design owns the banana palette")], des)
    _build_archive(tmp_path / "cp", [("p.txt", "company parent root memo")], parent)

    parent_only_docs = len(read_archive(parent).documents_)

    compose = runner.invoke(
        app,
        [
            "compose",
            str(parent),
            "--add",
            str(eng),
            "--name",
            "eng",
            "--add",
            str(des),
            "--name",
            "des",
        ],
    )
    assert compose.exit_code == 0, compose.output

    reloaded = read_archive(parent)
    assert [c.name for c in reloaded.manifest.children] == ["des", "eng"]  # stored sorted

    # Federated stats > parent-only (flatten merges self + both children).
    fed = KnowledgeSpace.load(str(parent))
    assert fed.stats.documents > parent_only_docs

    # A federated query reaches the child-only "apple" body.
    q = runner.invoke(app, ["query", "apple service", str(parent), "-k", "5"])
    assert q.exit_code == 0, q.output
    assert "apple" in q.output.lower()

    # --no-children limits to the parent: the child-only source rows vanish. (The query echo in
    # the table title always contains the query text, so query a parent word and assert the
    # child source files never appear in any hit row — only the parent's own p.txt does.)
    q_parent = runner.invoke(app, ["query", "memo", str(parent), "-k", "5", "--no-children"])
    assert q_parent.exit_code == 0, q_parent.output
    assert "p.txt" in q_parent.output
    assert "e.txt" not in q_parent.output
    assert "d.txt" not in q_parent.output


# ----------------------------------------------------------------- F3: CRUD on a sealed archive


def test_f3_add_update_rm_and_requery(tmp_path: Path) -> None:
    """Add a doc, update it, remove another — re-query reflects each mutation (F3)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "policy.txt").write_text("# Policy\n\npaid leave details.\n", encoding="utf-8")
    (src / "guide.txt").write_text("# Guide\n\nonboarding details.\n", encoding="utf-8")
    out = tmp_path / "out"
    built = runner.invoke(app, ["build", str(src), "--out", str(out), "--offline"])
    assert built.exit_code == 0, built.output
    archive = out / "handbook.indx"

    base_docs = len(read_archive(archive).documents_)

    # ADD a brand-new file → doc count grows, query finds its unique token.
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n", encoding="utf-8")
    added = runner.invoke(app, ["add", str(src / "extra.txt"), str(archive)])
    assert added.exit_code == 0, added.output
    assert "added 1 doc" in added.stdout
    assert len(read_archive(archive).documents_) == base_docs + 1
    found = runner.invoke(app, ["query", "zylophant", str(archive), "-k", "3"])
    assert found.exit_code == 0 and "zylophant" in found.stdout.lower()

    # UPDATE a file's content → the new token is retrievable.
    (src / "policy.txt").write_text("# Policy\n\nfrobnicate replaced text.\n", encoding="utf-8")
    updated = runner.invoke(app, ["update", str(src / "policy.txt"), str(archive)])
    assert updated.exit_code == 0, updated.output
    assert "updated 1 doc" in updated.stdout
    upd = runner.invoke(app, ["query", "frobnicate", str(archive), "-k", "3"])
    assert upd.exit_code == 0 and "frobnicate" in upd.stdout.lower()

    # RM a doc by path → count shrinks, its source no longer appears in hits.
    removed = runner.invoke(app, ["rm", "guide.txt", str(archive)])
    assert removed.exit_code == 0, removed.output
    assert "removed 1 doc" in removed.stdout
    after = read_archive(archive)
    assert all(d.path != "guide.txt" for d in after.documents_)
    gone = runner.invoke(app, ["query", "onboarding", str(archive), "-k", "5"])
    assert gone.exit_code == 0 and "guide.txt" not in gone.stdout

    # Relations stay consistent: no dangling edge names a removed doc/chunk id.
    live = {d.id for d in after.documents_} | {c.id for c in after.chunks}
    assert all(r.src in live and r.dst in live for r in after.relations)


# ----------------------------------------------------------------- F4: home DB (add→query→ask)


def test_f4_home_add_query_ask_with_no_space(tmp_path: Path) -> None:
    """$INDX_HOME: add→query→ask with no explicit space, plus home stats/reset (F4)."""
    # The isolated home starts empty.
    home_stats0 = runner.invoke(app, ["home", "stats", "--json"])
    assert home_stats0.exit_code == 0, home_stats0.output
    assert json.loads(home_stats0.stdout)["documents"] == 0

    # ADD a file to home with NO space argument → appends to the home space.
    note = tmp_path / "note.txt"
    note.write_text(
        "# Note\n\nOnboarding starts with a welcome session and a paid-leave overview.\n",
        encoding="utf-8",
    )
    added = runner.invoke(app, ["add", str(note)])
    assert added.exit_code == 0, added.output
    assert "added 1 doc" in added.stdout

    # home stats now reports the seeded doc (persisted across the process boundary of CliRunner).
    home_stats1 = runner.invoke(app, ["home", "stats", "--json"])
    assert json.loads(home_stats1.stdout)["documents"] == 1

    # QUERY home with NO space argument → finds the seeded content.
    queried = runner.invoke(app, ["query", "welcome onboarding", "-k", "3", "--json"])
    assert queried.exit_code == 0, queried.output
    payload = json.loads(queried.stdout)
    assert isinstance(payload, list) and payload

    # ASK home with NO space argument → offline extractive answer with cited Sources.
    asked = runner.invoke(app, ["ask", "how do I onboard?"])
    assert asked.exit_code == 0, asked.output
    assert "Sources" in asked.output

    # home path prints the isolated tmp dir (never the real ~/.indx).
    path_res = runner.invoke(app, ["home", "path"])
    assert path_res.exit_code == 0
    assert str(tmp_path / "home") in path_res.stdout

    # RESET (confirmed) wipes the home space back to empty.
    reset = runner.invoke(app, ["home", "reset", "--yes"])
    assert reset.exit_code == 0, reset.output
    home_stats2 = runner.invoke(app, ["home", "stats", "--json"])
    assert json.loads(home_stats2.stdout)["documents"] == 0


def test_f4_home_ask_sdk_offline_extractive(tmp_path: Path) -> None:
    """The SDK ``KnowledgeSpace.ask`` against the home space is offline + extractive (F4)."""
    note = tmp_path / "note.txt"
    note.write_text(
        "# Leave\n\nEmployees may take paid leave; requests go to the manager.\n", encoding="utf-8"
    )
    added = runner.invoke(app, ["add", str(note)])
    assert added.exit_code == 0, added.output

    from indx.home import open_home

    space = open_home()
    answer = space.ask("paid leave")
    assert isinstance(answer, Answer)
    assert answer.llm == "none"  # offline default, no model touched
    assert answer.hits
    # M1: source list lives in answer.sources, not embedded as a "Sources:" block in the body.
    assert "Sources:" not in answer.answer
    assert any("note.txt" in s for s in answer.sources)


def test_f4_home_add_does_not_persist_dangling_source_root(tmp_path: Path) -> None:
    """A home ``add`` of a stray file restores ``source_root`` to home, not a deleted temp dir.

    Regression for the staging temp dir leaking into the sealed home manifest: ``add`` of a single
    file stages it under a ``mkdtemp`` walk root, then removes that dir — the persisted manifest
    must not keep pointing at the now-deleted path.
    """
    from indx.home import home_dir, open_home

    note = tmp_path / "stray.txt"
    note.write_text("# Stray\n\na note about reimbursements.\n", encoding="utf-8")
    added = runner.invoke(app, ["add", str(note)])
    assert added.exit_code == 0, added.output

    source_root = Path(open_home().manifest.source_root)
    assert source_root.exists(), f"source_root {source_root} is dangling"
    assert source_root.resolve() == home_dir().resolve()
    assert "indx-home-add-" not in str(source_root)


def test_f1_fresh_build_suffix_omitting_walk_warns(tmp_path: Path) -> None:
    """A fresh ``build --from chunk`` omits the leading ``walk`` stage → stderr warning (F1).

    A single ``build`` has no prior state to resume from, so a suffix that skips ``walk``/``parse``
    indexes nothing; the user must be warned rather than silently get an empty archive.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# Alpha\n\nsome text.\n", encoding="utf-8")
    out = tmp_path / "out"
    result = runner.invoke(app, ["build", str(src), "--out", str(out), "--from", "chunk", *OFFLINE])
    assert result.exit_code == 0, result.output
    assert "omit the leading" in result.output and "walk" in result.output
    # The partial archive really is empty — the warning is not a false alarm.
    archive = out / "handbook.indx"
    assert read_archive(archive).documents_ == []
