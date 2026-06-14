"""Feature 2 — schema tolerance, ChildRef model, add/remove_child, children() resolution."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from indx import ChildRef, DirectoryPipeline, KnowledgeSpace
from indx.archive import format as fmt
from indx.archive import read_archive, write_archive
from indx.errors import ArchiveError


def _offline() -> DirectoryPipeline:
    return DirectoryPipeline(parser="plaintext", llm="none", embedder="hash", store="jsonl")


def _build_archive(src: Path, dest: Path) -> KnowledgeSpace:
    space = _offline().run(str(src))
    write_archive(space, dest)
    return space


def _corpus(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "note.txt").write_text(body, encoding="utf-8")
    return root


# ----------------------------------------------------------------- schema / reader tolerance


def test_schema_version_is_2() -> None:
    assert fmt.SCHEMA_VERSION == "2"
    assert "1" in fmt.SUPPORTED_SCHEMA_VERSIONS
    assert "2" in fmt.SUPPORTED_SCHEMA_VERSIONS


def test_fresh_archive_reports_v2(tmp_path: Path) -> None:
    _build_archive(_corpus(tmp_path / "c", "hello world"), tmp_path / "s.indx")
    loaded = read_archive(tmp_path / "s.indx")
    assert loaded.manifest.schema_version == "2"
    assert loaded.manifest.children == []


def test_v1_archive_still_loads(tmp_path: Path) -> None:
    """A hand-edited v1 manifest (no children key) loads; children defaults to []."""
    dest = tmp_path / "v1.indx"
    _build_archive(_corpus(tmp_path / "c", "alpha"), dest)
    _rewrite_manifest(dest, schema_version="1", drop_children=True)
    loaded = read_archive(dest)
    assert loaded.manifest.schema_version == "1"
    assert loaded.manifest.children == []


def test_unsupported_schema_version_raises(tmp_path: Path) -> None:
    dest = tmp_path / "v3.indx"
    _build_archive(_corpus(tmp_path / "c", "alpha"), dest)
    _rewrite_manifest(dest, schema_version="3", drop_children=False)
    with pytest.raises(ArchiveError, match="unsupported .indx schema_version"):
        read_archive(dest)


# ----------------------------------------------------------------- ChildRef model / round-trip


def test_childref_defaults_and_dump() -> None:
    ref = ChildRef(name="x", ref="x.indx")
    assert ref.sha256 is None
    assert ref.model_dump() == {"name": "x", "ref": "x.indx", "sha256": None}


def test_manifest_children_round_trip(tmp_path: Path) -> None:
    src = _corpus(tmp_path / "c", "alpha")
    space = _offline().run(str(src))
    space.add_child("x", "x.indx", sha256=None, auto_pin=False)
    dest = tmp_path / "p.indx"
    write_archive(space, dest)
    loaded = read_archive(dest)
    assert [c.model_dump() for c in loaded.manifest.children] == [
        {"name": "x", "ref": "x.indx", "sha256": None}
    ]


# ----------------------------------------------------------------- add_child / remove_child


def test_add_child_pins_and_sorts(tmp_path: Path) -> None:
    parent_src = _corpus(tmp_path / "p", "parent body")
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    _build_archive(_corpus(tmp_path / "ca", "a body"), a)
    _build_archive(_corpus(tmp_path / "cb", "b body"), b)
    parent = _build_archive(parent_src, tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")

    parent.add_child("zeta", "b.indx")
    parent.add_child("alpha", "a.indx")
    # sorted by name regardless of add order
    assert [c.name for c in parent.manifest.children] == ["alpha", "zeta"]
    # pinned (file resolvable)
    assert parent.manifest.children[0].sha256 == fmt.file_sha256(a)


def test_add_child_idempotent_replace(tmp_path: Path) -> None:
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("c", "first.indx", auto_pin=False)
    parent.add_child("c", "second.indx", auto_pin=False)
    assert len(parent.manifest.children) == 1
    assert parent.manifest.children[0].ref == "second.indx"


def test_remove_child(tmp_path: Path) -> None:
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent.add_child("c", "c.indx", auto_pin=False)
    removed = parent.remove_child("c")
    assert removed is not None and removed.name == "c"
    assert parent.manifest.children == []
    assert parent.remove_child("absent") is None


def test_add_child_does_not_touch_child_file(tmp_path: Path) -> None:
    a = tmp_path / "a.indx"
    _build_archive(_corpus(tmp_path / "ca", "a body"), a)
    before = a.read_bytes()
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("a", "a.indx")
    assert a.read_bytes() == before


def test_compose_reseal_is_byte_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    _build_archive(_corpus(tmp_path / "ca", "a body"), a)
    _build_archive(_corpus(tmp_path / "cb", "b body"), b)

    def compose(order: list[str], dest: Path) -> None:
        p = _build_archive(_corpus(tmp_path / "p", "parent"), dest)
        p._bind_source(dest)
        for nm in order:
            p.add_child(nm, f"{nm}.indx")
        p.save(str(dest))

    p1 = tmp_path / "p1.indx"
    p2 = tmp_path / "p2.indx"
    compose(["a", "b"], p1)
    compose(["b", "a"], p2)  # reverse add order -> identical manifest (sorted)
    assert p1.read_bytes() == p2.read_bytes()


# ----------------------------------------------------------------- children() resolution


def test_children_resolves_and_caches(tmp_path: Path) -> None:
    a = tmp_path / "a.indx"
    b = tmp_path / "b.indx"
    _build_archive(_corpus(tmp_path / "ca", "a body"), a)
    _build_archive(_corpus(tmp_path / "cb", "b body"), b)
    parent = _build_archive(_corpus(tmp_path / "p", "parent"), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("a", "a.indx")
    parent.add_child("b", "b.indx")
    parent.save(str(tmp_path / "p.indx"))

    reloaded = read_archive(tmp_path / "p.indx")
    kids = reloaded.children()
    assert len(kids) == 2
    assert kids is reloaded.children()  # cached identity


def test_children_missing_file_raises(tmp_path: Path) -> None:
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent.add_child("c", "nope.indx", auto_pin=False)
    parent.save(str(tmp_path / "p.indx"))
    reloaded = read_archive(tmp_path / "p.indx")
    with pytest.raises(ArchiveError, match="child archive not found"):
        reloaded.children()


def test_children_checksum_mismatch_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.indx"
    _build_archive(_corpus(tmp_path / "ca", "a body"), a)
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent._bind_source(tmp_path / "p.indx")
    parent.add_child("a", "a.indx")  # pins current bytes
    parent.save(str(tmp_path / "p.indx"))
    # Rewrite the child after pinning.
    _build_archive(_corpus(tmp_path / "ca", "TOTALLY DIFFERENT BODY now"), a)
    reloaded = read_archive(tmp_path / "p.indx")
    with pytest.raises(ArchiveError, match="child checksum mismatch"):
        reloaded.children()


def test_relative_ref_on_in_memory_space_raises(tmp_path: Path) -> None:
    space = _offline().run(str(_corpus(tmp_path / "c", "x")))  # never read from disk
    space.add_child("c", "c.indx", auto_pin=False)
    with pytest.raises(ArchiveError, match="no archive path"):
        space.children()


def test_corrupt_child_propagates(tmp_path: Path) -> None:
    bad = tmp_path / "bad.indx"
    bad.write_text("not a zip")
    parent = _build_archive(_corpus(tmp_path / "p", "x"), tmp_path / "p.indx")
    parent.add_child("bad", "bad.indx", auto_pin=False)
    parent.save(str(tmp_path / "p.indx"))
    reloaded = read_archive(tmp_path / "p.indx")
    with pytest.raises(ArchiveError):
        reloaded.children()


# ----------------------------------------------------------------- helpers


def _rewrite_manifest(dest: Path, *, schema_version: str, drop_children: bool) -> None:
    """Rewrite the manifest member of a .indx with a forced schema_version (test-only)."""
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
        blobs = {n: zf.read(n) for n in names}
    manifest = json.loads(blobs[fmt.MANIFEST].decode("utf-8"))
    manifest["schema_version"] = schema_version
    if drop_children:
        manifest.pop("children", None)
    new_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    blobs[fmt.MANIFEST] = new_text.encode("utf-8")
    # Re-stamp the checksum leg for the manifest so the reader's integrity check passes.
    record = json.loads(blobs[fmt.CHECKSUMS].decode("utf-8"))
    record["members"][fmt.MANIFEST] = fmt.checksum(new_text)
    blobs[fmt.CHECKSUMS] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n in names:
            zf.writestr(n, blobs[n])
