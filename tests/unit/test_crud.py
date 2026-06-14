"""Feature 3 — CRUD over indx documents (SDK mutators).

Exercises ``KnowledgeSpace.add``/``update``/``remove`` and the shared ``ingest_path`` helper on
the zero-dep offline core stack (parser=plaintext / llm=none / embedder=hash / store=jsonl /
output=.indx). Never touches torch/openai/qdrant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline, KnowledgeSpace


def _pipe() -> DirectoryPipeline:
    return DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output=".indx"
    )


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, Path]:
    """A 2-file offline-built archive plus its source dir. Returns (src, archive)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "policy.txt").write_text(
        "# Leave Policy\n\nEmployees may take paid leave.\n", encoding="utf-8"
    )
    (src / "guide.txt").write_text("# Onboarding\n\nWelcome to the team.\n", encoding="utf-8")
    space = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    archive = tmp_path / "space.indx"
    space.save(str(archive))
    return src, archive


def test_add_grows_counts(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    before = len(space.chunks)
    changed = space.add("extra.txt")
    assert len(space.documents()) == 3
    assert len(space.chunks) > before
    assert len(changed) == 1
    assert space.document(changed[0]) is not None
    assert space.document(changed[0]).path == "extra.txt"


def test_add_makes_doc_queryable(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nzylophant unique token.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    new_id = space.add("extra.txt")[0]
    hits = space.search("zylophant", k=5)
    assert any(h.chunk.doc_id == new_id for h in hits)


def test_remove_shrinks_and_drops_vectors(built: tuple[Path, Path]) -> None:
    _, archive = built
    space = KnowledgeSpace.load(str(archive))
    target = space.documents()[0]
    chunk_ids = {c.id for c in space.chunks_for(target.id)}
    removed = space.remove(target.path)
    assert removed == [target.id]
    assert space.document(target.id) is None
    assert not any(c.id in chunk_ids for c in space.chunks)
    # No relation names the removed doc.
    assert all(r.src != target.id and r.dst != target.id for r in space.relations)


def test_remove_by_path_and_by_id(built: tuple[Path, Path]) -> None:
    _, archive = built
    space = KnowledgeSpace.load(str(archive))
    by_path = space.documents()[0]
    assert space.remove(by_path.path) == [by_path.id]

    space2 = KnowledgeSpace.load(str(archive))
    by_id = space2.documents()[0]
    assert space2.remove(by_id.id) == [by_id.id]

    # Non-existent target is a clean no-op.
    space3 = KnowledgeSpace.load(str(archive))
    docs_before = len(space3.documents_)
    assert space3.remove("does/not/exist.txt") == []
    assert len(space3.documents_) == docs_before


def test_update_changes_content(built: tuple[Path, Path]) -> None:
    src, archive = built
    space = KnowledgeSpace.load(str(archive))
    target = next(d for d in space.documents() if d.path == "policy.txt")
    old_text = " ".join(c.text for c in space.chunks_for(target.id))
    assert "paid leave" in old_text

    (src / "policy.txt").write_text(
        "# Leave Policy\n\nfrobnicate widget replaced content.\n", encoding="utf-8"
    )
    changed = space.update("policy.txt")
    assert changed == [target.id]  # same stable id
    new_chunks = space.chunks_for(target.id)
    assert any("frobnicate" in c.text for c in new_chunks)
    assert all("paid leave" not in c.text for c in new_chunks)
    # The embeddings were re-derived: a search for the new text returns the updated chunk.
    hits = space.search("frobnicate widget", k=5)
    assert any("frobnicate" in h.chunk.text for h in hits)


def test_readd_is_idempotent(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nstable content.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    space.add("extra.txt")
    docs_after_one = len(space.documents_)
    chunks_after_one = len(space.chunks)
    ids_after_one = sorted(d.id for d in space.documents_)

    space.add("extra.txt")
    assert len(space.documents_) == docs_after_one
    assert len(space.chunks) == chunks_after_one
    assert sorted(d.id for d in space.documents_) == ids_after_one
    # No duplicate document for the path.
    assert sum(1 for d in space.documents_ if d.path == "extra.txt") == 1


def test_relations_stay_consistent(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "team").mkdir(parents=True)
    (src / "team" / "a.txt").write_text("# A\n\nalpha.\n", encoding="utf-8")
    (src / "team" / "b.txt").write_text("# B\n\nbravo.\n", encoding="utf-8")
    space = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    archive = tmp_path / "s.indx"
    space.save(str(archive))

    # Add a sibling — it should create sibling edges to the existing folder-mates.
    (src / "team" / "c.txt").write_text("# C\n\ncharlie.\n", encoding="utf-8")
    loaded = KnowledgeSpace.load(str(archive))
    new_id = loaded.add("team/c.txt")[0]
    assert any(r.type == "sibling" and new_id in (r.src, r.dst) for r in loaded.relations)
    # Remove one and confirm no edge names it and nothing dangles.
    loaded.remove("team/a.txt")
    live = {d.id for d in loaded.documents_} | {c.id for c in loaded.chunks}
    assert all(r.src in live and r.dst in live for r in loaded.relations)


def test_chunk_source_restamped(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nnote.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    new_id = space.add("extra.txt")[0]
    doc = space.document(new_id)
    assert doc is not None
    for chunk in space.chunks_for(new_id):
        assert chunk.source is not None
        assert chunk.source.path == doc.path
        assert chunk.source.folder == doc.folder


def test_archive_loaded_space_has_no_store_but_crud_works(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nnote.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    # A freshly loaded space has no live pipeline store; add must rebuild a jsonl store.
    changed = space.add("extra.txt")
    assert changed
    # Vectors present so search works.
    assert space.search("note", k=3)


def test_save_load_round_trip_after_crud_is_byte_deterministic(built: tuple[Path, Path]) -> None:
    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nnote.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    space.add("extra.txt")
    a = src.parent / "a.indx"
    b = src.parent / "b.indx"
    space.save(str(a))
    space.save(str(b))
    assert a.read_bytes() == b.read_bytes()


def test_build_then_add_equals_full_build(tmp_path: Path) -> None:
    """Build a,b,c full vs build a,b (same source_root) then add c → byte-identical members."""
    src = tmp_path / "src"
    src.mkdir()
    for n in ("a", "b", "c"):
        (src / f"{n}.txt").write_text(f"# {n}\n\n{n} content here.\n", encoding="utf-8")

    full = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    full_archive = tmp_path / "full.indx"
    full.save(str(full_archive))

    # Same source_root: build everything, remove c, then add c back.
    partial = _pipe().run(str(src), str(tmp_path / "o2.jsonl"))
    partial.remove("c.txt")
    part_archive = tmp_path / "part.indx"
    partial.save(str(part_archive))
    loaded = KnowledgeSpace.load(str(part_archive))
    loaded.add("c.txt")
    loaded.save(str(part_archive))

    assert full_archive.read_bytes() == part_archive.read_bytes()


def test_add_path_outside_root_raises(built: tuple[Path, Path]) -> None:
    from indx.errors import ArchiveError

    _, archive = built
    space = KnowledgeSpace.load(str(archive))
    with pytest.raises(ArchiveError):
        space.add("/etc/hostname")


def _relative_root_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a space whose ``manifest.source_root`` is RELATIVE, save it, return the archive path.

    Mirrors the CLI: ``indx ./src`` records ``source_root='src'`` (relative). The CWD is moved into
    ``tmp_path`` so a cwd-relative ``src/...`` add/remove resolves the way the CLI PATH arg does.
    """
    src = tmp_path / "src"
    (src / "zzz").mkdir(parents=True)
    (src / "zzz" / "d.txt").write_text("# D\n\ndelta body.\n", encoding="utf-8")
    space = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    space.manifest.source_root = "src"  # force the relative form the CLI stores
    archive = tmp_path / "rel.indx"
    space.save(str(archive))
    monkeypatch.chdir(tmp_path)
    return archive


def test_add_cwd_relative_path_with_relative_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug #4: ``add('src/zzz/added.txt')`` (cwd-relative, relative source_root) must add 1 doc."""
    archive = _relative_root_space(tmp_path, monkeypatch)
    (tmp_path / "src" / "zzz" / "added.txt").write_text(
        "# Added\n\nzylophant token.\n", encoding="utf-8"
    )
    space = KnowledgeSpace.load(str(archive))
    changed = space.add("src/zzz/added.txt")  # the SAME string the CLI passes
    assert len(changed) == 1
    doc = space.document(changed[0])
    assert doc is not None
    assert doc.path == "zzz/added.txt"  # stored root-relative, not double-joined


def test_add_source_root_relative_path_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK source-root-relative convention (``add('zzz/added.txt')``) keeps working too."""
    archive = _relative_root_space(tmp_path, monkeypatch)
    (tmp_path / "src" / "zzz" / "added.txt").write_text("# Added\n\nbody.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    changed = space.add("zzz/added.txt")
    assert len(changed) == 1
    assert space.document(changed[0]).path == "zzz/added.txt"


def test_remove_cwd_relative_path_with_source_root_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug #S5: ``remove('src/zzz/d.txt')`` (cwd-relative) must drop the doc, dir present."""
    archive = _relative_root_space(tmp_path, monkeypatch)
    space = KnowledgeSpace.load(str(archive))
    removed = space.remove("src/zzz/d.txt")
    assert len(removed) == 1
    assert space.document(removed[0]) is None
    assert not any(d.path == "zzz/d.txt" for d in space.documents_)


def test_remove_works_when_source_root_dir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug #S5: removal must work even when the source_root dir no longer exists on disk."""
    archive = _relative_root_space(tmp_path, monkeypatch)
    # Rename the source dir away so ``root.is_dir()`` is False.
    (tmp_path / "src").rename(tmp_path / "src-moved")

    space = KnowledgeSpace.load(str(archive))
    removed = space.remove("src/zzz/d.txt")  # cwd-relative add-style path
    assert len(removed) == 1
    assert space.document(removed[0]) is None

    # The bare stored path also still resolves with the dir absent.
    space2 = KnowledgeSpace.load(str(archive))
    assert len(space2.remove("zzz/d.txt")) == 1


def _xref_space(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """Build a 2-doc space where a.txt references b.txt. Returns (src, archive, a_id, b_id)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("# A\n\nSee b.txt for details.\n", encoding="utf-8")
    (src / "b.txt").write_text("# B\n\nbravo body.\n", encoding="utf-8")
    space = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    archive = tmp_path / "xref.indx"
    space.save(str(archive))
    a_id = next(d.id for d in space.documents_ if d.path == "a.txt")
    b_id = next(d.id for d in space.documents_ if d.path == "b.txt")
    # The full build produced the references edge.
    assert any(
        str(r.type) == "references" and r.src == a_id and r.dst == b_id for r in space.relations
    )
    return src, archive, a_id, b_id


def test_reference_edge_survives_update_of_target(tmp_path: Path) -> None:
    """Bug #6: updating the referenced doc (b) must not destroy the a->b references edge."""
    _, archive, a_id, b_id = _xref_space(tmp_path)
    space = KnowledgeSpace.load(str(archive))
    space.update("b.txt")  # b is in new_ids; the a->b edge merely TOUCHES it
    assert any(
        str(r.type) == "references" and r.src == a_id and r.dst == b_id for r in space.relations
    )
    live = {d.id for d in space.documents_}
    assert all(r.src in live and r.dst in live for r in space.relations)


def test_reference_edge_survives_readd_of_referencer(tmp_path: Path) -> None:
    """Bug #6: re-adding the referencer (a) recomputes its outgoing edge, not lose it."""
    src, archive, a_id, b_id = _xref_space(tmp_path)
    space = KnowledgeSpace.load(str(archive))
    space.add("a.txt")  # re-add the source of the edge
    assert any(
        str(r.type) == "references" and r.src == a_id and r.dst == b_id for r in space.relations
    )
    live = {d.id for d in space.documents_}
    assert all(r.src in live and r.dst in live for r in space.relations)


def test_add_embedder_dim_mismatch_raises_and_does_not_corrupt(built: tuple[Path, Path]) -> None:
    """Bug #S3: adding with a different-width embedder must raise, not silently mix widths."""
    from indx.errors import ArchiveError

    src, archive = built
    (src / "extra.txt").write_text("# Extra\n\nnote.\n", encoding="utf-8")
    space = KnowledgeSpace.load(str(archive))
    # Force the space to claim a different embedder/width than the offline hash add will produce.
    space.manifest.embedding_model = "other"
    space.manifest.embedding_dim = 1536
    widths_before = {len(c.embedding) for c in space.chunks if c.embedding}
    docs_before = len(space.documents_)

    with pytest.raises(ArchiveError, match="embedder mismatch"):
        space.add("extra.txt")

    # No wrong-width chunks were appended and no doc rows leaked.
    assert len(space.documents_) == docs_before
    widths_after = {len(c.embedding) for c in space.chunks if c.embedding}
    assert widths_after == widths_before


def test_ingest_path_single_file(tmp_path: Path) -> None:
    from indx.pipeline.pipeline import ingest_path
    from indx.registry import get_store
    from indx.store.base import VectorStore

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("# A\n\nalpha body.\n", encoding="utf-8")
    (src / "b.txt").write_text("# B\n\nbravo body.\n", encoding="utf-8")

    store = get_store("jsonl")
    assert isinstance(store, VectorStore)
    result = ingest_path(_pipe(), src, ["a.txt"], store=store)
    assert [d.path for d in result.documents] == ["a.txt"]
    assert result.chunks
    assert all(c.doc_id == result.documents[0].id for c in result.chunks)
    # The provided store received the vectors.
    assert store.search(result.chunks[0].embedding or [], k=1)


def test_ingest_path_full_matches_build(tmp_path: Path) -> None:
    """ingest_path(..., None) reproduces a full build's documents/chunks/relations."""
    from indx.pipeline.pipeline import ingest_path

    src = tmp_path / "src"
    src.mkdir()
    for n in ("a", "b"):
        (src / f"{n}.txt").write_text(f"# {n}\n\n{n} body.\n", encoding="utf-8")

    built = _pipe().run(str(src), str(tmp_path / "o.jsonl"))
    result = ingest_path(_pipe(), src, None)
    assert [d.id for d in result.documents] == [d.id for d in built.documents_]
    assert [c.id for c in result.chunks] == [c.id for c in built.chunks]
    assert len(result.relations) == len(built.relations)
