"""Feature 1 — granular per-stage load / import.

All-offline (parser=plaintext, llm=none, embedder=hash, store=jsonl, output=.indx) coverage of:

* selective stage execution — ``DirectoryPipeline.run(stages=...)`` + ``--stages``/``--through``/
  ``--from`` (unknown stage → RegistryError exit 3; usage conflicts → exit 2);
* selective archive load — ``read_archive(members=...)`` / ``KnowledgeSpace.load(members=...)`` /
  ``KnowledgeSpace.load_part`` (guards follow what is read);
* CLI ``inspect --part``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx import DirectoryPipeline, KnowledgeSpace
from indx.archive import format as fmt
from indx.archive import read_archive
from indx.archive.reader import MAX_MEMBER_BYTES
from indx.cli.app import app
from indx.cli.build import _stage_subset
from indx.errors import ArchiveError, RegistryError

runner = CliRunner()

OFFLINE = ["--parser", "plaintext", "--llm", "none", "--embedder", "hash", "--store", "jsonl"]


def _corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "notes").mkdir(parents=True, exist_ok=True)
    (src / "notes" / "alpha.md").write_text("# Alpha\n\nOnboarding handbook lives here.\n")
    (src / "beta.md").write_text("# Beta\n\nSecurity policy details and references.\n")
    return src


def _offline_pipeline() -> DirectoryPipeline:
    return DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output="indx"
    )


# ----------------------------------------------------------------- selective stage execution


def test_run_stages_walk_only(tmp_path: Path) -> None:
    space = _offline_pipeline().run(_corpus(tmp_path))
    walked = _offline_pipeline().run(_corpus(tmp_path), stages=("walk",))
    assert len(walked.documents_) == len(space.documents_) > 0
    assert walked.chunks == []
    assert walked.relations == []
    assert walked.manifest.embedding_dim is None


def test_run_through_chunk(tmp_path: Path) -> None:
    space = _offline_pipeline().run(_corpus(tmp_path), stages=("walk", "parse", "chunk"))
    assert len(space.documents_) > 0
    assert len(space.chunks) > 0
    assert space.relations == []
    assert space.manifest.embedding_dim is None
    assert space.stats.embeddings == 0


def test_run_stages_order_independent(tmp_path: Path) -> None:
    """Caller order does not matter — canonical order is used."""
    a = _offline_pipeline().run(_corpus(tmp_path), stages=("walk", "parse", "chunk"))
    b = _offline_pipeline().run(_corpus(tmp_path), stages=("chunk", "walk", "parse"))
    assert [d.path for d in a.documents_] == [d.path for d in b.documents_]
    assert [c.id for c in a.chunks] == [c.id for c in b.chunks]


def test_partial_run_seals_loadable_archive(tmp_path: Path) -> None:
    out = tmp_path / "partial.indx"
    _offline_pipeline().run(_corpus(tmp_path), out, stages=("walk", "parse", "chunk"))
    loaded = read_archive(out)
    assert len(loaded.documents_) > 0
    assert len(loaded.chunks) > 0
    assert loaded.stats.embeddings == 0


def test_run_unknown_stage_raises_registry_error(tmp_path: Path) -> None:
    out = tmp_path / "nope.indx"
    with pytest.raises(RegistryError) as exc:
        _offline_pipeline().run(_corpus(tmp_path), out, stages=("walk", "nope"))
    assert "nope" in str(exc.value)
    assert "walk" in str(exc.value)  # lists known stages
    assert not out.exists()  # nothing written before the failure


def test_run_unknown_stage_resolves_against_live_list(tmp_path: Path) -> None:
    pipe = _offline_pipeline()
    pipe.drop("enrich")
    with pytest.raises(RegistryError):
        pipe.run(_corpus(tmp_path), stages=("enrich",))


# ----------------------------------------------------------------- _stage_subset helper


_STAGES = ["walk", "parse", "chunk", "relate", "enrich", "embed-pack"]


def test_stage_subset_none_when_no_flags() -> None:
    assert _stage_subset(_STAGES, stages=None, through=None, from_stage=None) is None


def test_stage_subset_explicit_list_canonical_order() -> None:
    assert _stage_subset(_STAGES, stages="chunk,walk", through=None, from_stage=None) == [
        "walk",
        "chunk",
    ]


def test_stage_subset_through_prefix() -> None:
    assert _stage_subset(_STAGES, stages=None, through="chunk", from_stage=None) == [
        "walk",
        "parse",
        "chunk",
    ]


def test_stage_subset_from_suffix() -> None:
    assert _stage_subset(_STAGES, stages=None, through=None, from_stage="relate") == [
        "relate",
        "enrich",
        "embed-pack",
    ]


def test_stage_subset_window() -> None:
    assert _stage_subset(_STAGES, stages=None, through="enrich", from_stage="relate") == [
        "relate",
        "enrich",
    ]


def test_stage_subset_conflict_raises() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _stage_subset(_STAGES, stages="walk", through="chunk", from_stage=None)


def test_stage_subset_empty_window_raises() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _stage_subset(_STAGES, stages=None, through="walk", from_stage="embed-pack")


def test_stage_subset_unknown_through_raises() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        _stage_subset(_STAGES, stages=None, through="bogus", from_stage=None)


# ----------------------------------------------------------------- CLI build flags


def test_cli_through_chunk_no_embeddings(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--through", "chunk", *OFFLINE]
    )
    assert result.exit_code == 0, result.stdout
    inspected = runner.invoke(app, ["inspect", str(out), "--json"])
    stats = json.loads(inspected.stdout)
    assert stats["chunks"] > 0
    assert stats["embeddings"] == 0


def test_cli_from_relate_clean_run(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--from", "relate", *OFFLINE]
    )
    assert result.exit_code == 0, result.stdout


def test_cli_stages_and_through_byte_identical(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    r1 = runner.invoke(
        app, ["build", str(src), "--out", str(out1), "--stages", "walk,parse,chunk", *OFFLINE]
    )
    r2 = runner.invoke(app, ["build", str(src), "--out", str(out2), "--through", "chunk", *OFFLINE])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert (out1 / "handbook.indx").read_bytes() == (out2 / "handbook.indx").read_bytes()


def test_cli_stages_and_through_conflict_exit_2(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["build", str(src), "--out", str(out), "--stages", "walk", "--through", "chunk", *OFFLINE],
    )
    assert result.exit_code == 2


def test_cli_empty_window_exit_2(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    args = ["build", str(src), "--out", str(out), "--from", "embed-pack", "--through", "walk"]
    result = runner.invoke(app, [*args, *OFFLINE])
    assert result.exit_code == 2


def test_cli_unknown_stage_exit_3(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--stages", "bogus", *OFFLINE]
    )
    assert result.exit_code == 3
    assert "Traceback" not in result.stdout


# ----------------------------------------------------------------- selective archive load


def _build_archive(tmp_path: Path) -> Path:
    out = tmp_path / "full.indx"
    _offline_pipeline().run(_corpus(tmp_path), out)
    return out


def test_read_archive_subset_documents_only(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    space = read_archive(p, members=("documents",))
    assert len(space.documents_) > 0
    assert space.chunks == []
    assert space.relations == []


def test_read_archive_none_equals_full(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    full = read_archive(p)
    subset = read_archive(p, members=None)
    assert len(subset.documents_) == len(full.documents_)
    assert len(subset.chunks) == len(full.chunks)
    assert len(subset.relations) == len(full.relations)


def test_knowledge_space_load_members_and_load_part(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    space = KnowledgeSpace.load(str(p), members=("chunks",))
    assert len(space.chunks) > 0
    assert space.documents_ == []
    rels = KnowledgeSpace.load_part(str(p), "relations")
    from indx.core.relation import Relation

    assert all(isinstance(r, Relation) for r in rels)


def test_read_archive_unknown_member_raises(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    with pytest.raises(ArchiveError):
        read_archive(p, members=("frobnicate",))
    with pytest.raises(ArchiveError):
        KnowledgeSpace.load_part(str(p), "manifest")


def _tamper_member(src: Path, dst: Path, member: str, payload: bytes) -> None:
    """Copy ``src`` to ``dst`` replacing one member's bytes (corrupting its checksum)."""
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    data[member] = payload
    with zipfile.ZipFile(dst, "w") as zout:
        for n in names:
            zout.writestr(n, data[n])


def test_checksum_guard_follows_read_members(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    tampered = tmp_path / "tampered.indx"
    _tamper_member(p, tampered, fmt.CHUNKS, b'{"id":"x"}\n')
    # Reading the tampered (chunks) member → checksum mismatch.
    with pytest.raises(ArchiveError):
        read_archive(tampered, members=("chunks",))
    # Reading a DIFFERENT member on the same archive succeeds (tampered member never read).
    docs_only = read_archive(tampered, members=("documents",))
    assert len(docs_only.documents_) > 0


def test_zip_bomb_guard_only_on_read_members(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    bomb = tmp_path / "bomb.indx"
    huge = b"x" * (MAX_MEMBER_BYTES + 8)
    _tamper_member(p, bomb, fmt.RELATIONS, huge)
    with pytest.raises(ArchiveError):
        read_archive(bomb, members=("relations",))
    # Excluding the oversized member → no inflation, clean load.
    safe = read_archive(bomb, members=("documents",))
    assert len(safe.documents_) > 0


def test_schema_guard_always_runs(tmp_path: Path) -> None:
    p = _build_archive(tmp_path)
    bad = tmp_path / "badschema.indx"
    with zipfile.ZipFile(p) as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    manifest = json.loads(data[fmt.MANIFEST])
    manifest["schema_version"] = "999"
    data[fmt.MANIFEST] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(bad, "w") as zout:
        for n in names:
            zout.writestr(n, data[n])
    with pytest.raises(ArchiveError):
        read_archive(bad, members=("documents",))


# ----------------------------------------------------------------- CLI inspect --part


def test_inspect_part_documents_json(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0
    full = read_archive(out / "handbook.indx")
    result = runner.invoke(app, ["inspect", str(out), "--part", "documents", "--json"])
    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert len(rows) == len(full.documents_)


def test_inspect_part_chunks_and_relations_json(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0
    full = read_archive(out / "handbook.indx")
    rc = runner.invoke(app, ["inspect", str(out), "--part", "chunks", "--json"])
    rr = runner.invoke(app, ["inspect", str(out), "--part", "relations", "--json"])
    assert rc.exit_code == 0 and rr.exit_code == 0
    assert len(json.loads(rc.stdout)) == len(full.chunks)
    assert len(json.loads(rr.stdout)) == len(full.relations)


def test_inspect_part_manifest_json(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0
    result = runner.invoke(app, ["inspect", str(out), "--part", "manifest", "--json"])
    assert result.exit_code == 0, result.stdout
    obj = json.loads(result.stdout)
    assert "schema_version" in obj
    assert "indx_version" in obj
    assert "components" in obj


def test_inspect_part_bogus_exit_2(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0
    result = runner.invoke(app, ["inspect", str(out), "--part", "bogus"])
    assert result.exit_code == 2


def test_inspect_part_with_documents_mutually_exclusive(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    assert runner.invoke(app, ["build", str(src), "--out", str(out), *OFFLINE]).exit_code == 0
    result = runner.invoke(app, ["inspect", str(out), "--part", "documents", "--documents", "x"])
    assert result.exit_code == 2


def test_inspect_part_documents_on_jsonl_dir(tmp_path: Path) -> None:
    src = _corpus(tmp_path)
    out = tmp_path / "out"
    built = runner.invoke(
        app, ["build", str(src), "--out", str(out), "--format", "jsonl", *OFFLINE]
    )
    assert built.exit_code == 0, built.stdout
    result = runner.invoke(app, ["inspect", str(out), "--part", "documents", "--json"])
    assert result.exit_code == 0, result.stdout
    assert isinstance(json.loads(result.stdout), list)
