"""T1 — individual pipeline stages on synthetic input (no heavy slots)."""

from __future__ import annotations

from indx.core.relation import RelationType
from indx.embed.hash_embedder import HashEmbedder
from indx.parsers.plaintext import PlainTextParser
from indx.pipeline.stages import (
    ChunkStage,
    EnrichStage,
    PackStage,
    ParseStage,
    RelateStage,
    WalkStage,
)
from indx.pipeline.stages.chunk import _pack
from indx.store.jsonl import JsonlStore


def _walked(ctx):
    return WalkStage().run(ctx)


def test_walk_captures_lineage_sorted(empty_context, tmp_tree) -> None:
    empty_context.root = tmp_tree
    ctx = _walked(empty_context)
    paths = [d.path for d in ctx.space.documents_]
    assert paths == sorted(paths)  # deterministic order
    a_doc = next(d for d in ctx.space.documents_ if d.path.endswith("one.md"))
    assert a_doc.lineage == ["a"]


def test_chunk_links_neighbors(empty_context, tmp_tree) -> None:
    empty_context.root = tmp_tree
    ctx = PackStage(HashEmbedder(), JsonlStore()).run(
        EnrichStage().run(
            ChunkStage().run(ParseStage(PlainTextParser()).run(_walked(empty_context)))
        )
    )
    for chunks in (ctx.space.chunks_for(d.id) for d in ctx.space.documents_):
        for i, c in enumerate(chunks):
            assert (c.prev_id is None) == (i == 0)
            assert (c.next_id is None) == (i == len(chunks) - 1)


def test_relate_sibling_and_continues(empty_context, tmp_path) -> None:
    (tmp_path / "report-part-1.txt").write_text("intro\n")
    (tmp_path / "report-part-2.txt").write_text("body\n")
    empty_context.root = tmp_path
    ctx = RelateStage().run(_walked(empty_context))
    types = {r.type for r in ctx.space.relations}
    assert RelationType.SIBLING in types
    assert RelationType.CONTINUES in types


def test_pack_hard_splits_oversized_paragraph() -> None:
    # A single paragraph that has no \n\n boundary must still be bounded by max_chars.
    pieces = _pack("x" * 2000, 800)
    assert all(len(p) <= 800 for p in pieces)
    assert "".join(pieces) == "x" * 2000  # deterministic, lossless reassembly
    assert pieces == ["x" * 800, "x" * 800, "x" * 400]


def test_pack_respects_max_chars_after_packing() -> None:
    # Packed multi-paragraph pieces are also capped to max_chars.
    text = "\n\n".join(["y" * 500, "z" * 500])
    assert all(len(p) <= 300 for p in _pack(text, 300))


def test_pack_sets_embeddings_and_manifest(empty_context, tmp_tree) -> None:
    empty_context.root = tmp_tree
    ctx = PackStage(HashEmbedder(dim=64), JsonlStore()).run(
        ChunkStage().run(ParseStage(PlainTextParser()).run(_walked(empty_context)))
    )
    assert ctx.space.manifest.embedding_dim == 64
    assert all(c.embedding is not None and len(c.embedding) == 64 for c in ctx.space.chunks)
