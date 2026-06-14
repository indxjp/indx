"""Feature 2 — flatten() semantics, cycle guard, federated stats/search."""

from __future__ import annotations

from pathlib import Path

from indx import DirectoryPipeline, KnowledgeSpace
from indx.archive import read_archive, write_archive


def _offline() -> DirectoryPipeline:
    return DirectoryPipeline(parser="plaintext", llm="none", embedder="hash", store="jsonl")


def _corpus(root: Path, *files: tuple[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files:
        (root / name).write_text(body, encoding="utf-8")
    return root


def _build(src: Path, dest: Path) -> KnowledgeSpace:
    space = _offline().run(str(src))
    write_archive(space, dest)
    return space


def _composite(tmp_path: Path) -> KnowledgeSpace:
    """parent (1 file) referencing two children (1 file each), composed + reloaded."""
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    _build(_corpus(tmp_path / "ca", ("a.txt", "alpha apple onboarding")), a)
    _build(_corpus(tmp_path / "cb", ("b.txt", "beta banana payroll")), b)
    parent = _build(_corpus(tmp_path / "cp", ("p.txt", "parent root memo")), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("eng", "a.indx")
    parent.add_child("design", "b.indx")
    parent.save(str(tmp_path / "p.indx"))
    return read_archive(tmp_path / "p.indx")


def test_flatten_merges_and_namespaces(tmp_path: Path) -> None:
    parent = _composite(tmp_path)
    kids = parent.children()
    merged = parent.flatten()

    expected_docs = len(parent.documents_) + sum(len(k.documents_) for k in kids)
    assert len(merged.documents_) == expected_docs
    assert len(merged.chunks) == len(parent.chunks) + sum(len(k.chunks) for k in kids)

    ids = {d.id for d in merged.documents_}
    # parent ids bare; child ids namespaced by ChildRef.name (sorted: design, eng)
    assert any("/" not in i for i in ids)
    assert any(i.startswith("design/") for i in ids)
    assert any(i.startswith("eng/") for i in ids)
    # merged view does not re-federate
    assert merged.manifest.children == []


def test_flatten_keeps_chunk_links_consistent(tmp_path: Path) -> None:
    parent = _composite(tmp_path)
    merged = parent.flatten()
    chunk_ids = {c.id for c in merged.chunks}
    doc_ids = {d.id for d in merged.documents_}
    for c in merged.chunks:
        assert c.doc_id in doc_ids
        if c.prev_id is not None:
            assert c.prev_id in chunk_ids
        if c.next_id is not None:
            assert c.next_id in chunk_ids
    for r in merged.relations:
        # src/dst point at namespaced docs/chunks present in the merge
        assert r.src in doc_ids or r.src in chunk_ids
        assert r.dst in doc_ids or r.dst in chunk_ids


def test_flatten_mutates_nothing_on_disk(tmp_path: Path) -> None:
    parent = _composite(tmp_path)
    p_bytes = (tmp_path / "p.indx").read_bytes()
    a_bytes = (tmp_path / "a.indx").read_bytes()
    parent.flatten()
    assert (tmp_path / "p.indx").read_bytes() == p_bytes
    assert (tmp_path / "a.indx").read_bytes() == a_bytes


def test_same_archive_referenced_twice_is_deduped(tmp_path: Path) -> None:
    a = tmp_path / "a.indx"
    _build(_corpus(tmp_path / "ca", ("a.txt", "shared body")), a)
    parent = _build(_corpus(tmp_path / "cp", ("p.txt", "root")), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("first", "a.indx")
    parent.add_child("second", "a.indx")  # same file, different name
    parent.save(str(tmp_path / "p.indx"))
    reloaded = read_archive(tmp_path / "p.indx")
    merged = reloaded.flatten()
    # visited-set keys on the real path: the shared archive's docs appear once.
    child_docs = read_archive(a).documents_
    assert len(merged.documents_) == len(reloaded.documents_) + len(child_docs)


def test_cycle_is_guarded(tmp_path: Path) -> None:
    """A->B->A terminates; each archive's docs appear once."""
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    _build(_corpus(tmp_path / "ca", ("a.txt", "alpha")), a)
    _build(_corpus(tmp_path / "cb", ("b.txt", "beta")), b)
    # Build a cyclic ref graph; use unpinned refs since each save() mutates the file bytes
    # (a stale sha256 pin would otherwise fire before the cycle guard is exercised).
    sa = read_archive(a)
    sa.add_child("b", "b.indx", auto_pin=False)
    sa.save(str(a))
    sb = read_archive(b)
    sb.add_child("a", "a.indx", auto_pin=False)
    sb.save(str(b))

    parent = _build(_corpus(tmp_path / "cp", ("p.txt", "root")), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("a", "a.indx", auto_pin=False)
    parent.save(str(tmp_path / "p.indx"))
    reloaded = read_archive(tmp_path / "p.indx")
    merged = reloaded.flatten()  # must terminate
    # parent + a + b, each once
    assert len(merged.documents_) == 3


def test_federated_stats(tmp_path: Path) -> None:
    parent = _composite(tmp_path)
    kids = parent.children()
    expected_docs = len(parent.documents_) + sum(len(k.documents_) for k in kids)
    assert parent.stats.documents == expected_docs
    assert parent.stats == parent.flatten().stats


def test_childless_stats_unchanged(tmp_path: Path) -> None:
    space = _build(_corpus(tmp_path / "c", ("x.txt", "plain")), tmp_path / "s.indx")
    assert space.stats.documents == len(space.documents_)


def test_federated_search_returns_child_hits(tmp_path: Path) -> None:
    parent = _composite(tmp_path)
    # "payroll" only lives in the design child.
    hits = parent.search("payroll", k=5)
    assert hits
    texts = " ".join(h.chunk.text for h in hits)
    assert "payroll" in texts
    # the winning chunk id is namespaced under design/
    assert any(h.chunk.id.startswith("design/") for h in hits)


def test_no_children_search_excludes_children(tmp_path: Path) -> None:
    from indx.cli._render import parent_only

    parent = _composite(tmp_path)
    only = parent_only(parent)
    hits = only.search("payroll", k=5)
    # payroll lives only in a child; parent-only must not surface it.
    assert all("payroll" not in h.chunk.text for h in hits)
