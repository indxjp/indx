"""S4 — IndxWriter embeddings/ row→chunk-id alignment.

``vectors.f32`` packs only embedded chunks, so once any chunk is unembedded the row index
no longer equals the chunk index. The embeddings manifest must therefore carry a row-aligned
``ids`` list so a reader can recover which chunk each vector row belongs to.
"""

from __future__ import annotations

import array
import json
from pathlib import Path

from indx.core import Chunk, Document, KnowledgeSpace
from indx.core.knowledge_space import Manifest
from indx.output.indx_writer import IndxWriter


def _space(embeddings: list[list[float] | None]) -> KnowledgeSpace:
    chunks = [
        Chunk(id=f"c{i}", doc_id="d1", position=i, text=f"chunk {i}", embedding=emb)
        for i, emb in enumerate(embeddings)
    ]
    return KnowledgeSpace(
        documents=[Document(id="d1", path="a.md")],
        chunks=chunks,
        manifest=Manifest(embedding_model="hash", embedding_dim=2, components={"store": "jsonl"}),
    )


def _read_rows(dest: Path, dim: int) -> list[list[float]]:
    buf = array.array("f")
    buf.frombytes((dest / "embeddings" / "vectors.f32").read_bytes())
    flat = buf.tolist()
    return [flat[i : i + dim] for i in range(0, len(flat), dim)]


def test_unembedded_middle_chunk_keeps_row_to_id_map(tmp_path: Path) -> None:
    # Middle chunk has no embedding; outer two carry distinct 2-dim vectors.
    space = _space([[1.0, 2.0], None, [3.0, 4.0]])
    IndxWriter().write(space, tmp_path)

    manifest = json.loads((tmp_path / "embeddings" / "manifest.json").read_text())
    assert manifest["count"] == 2  # only embedded chunks contribute rows
    assert manifest["ids"] == ["c0", "c2"]  # row order, skipping the unembedded c1
    assert len(manifest["ids"]) == manifest["count"]

    rows = _read_rows(tmp_path, manifest["dim"])
    assert len(rows) == manifest["count"]
    by_id = dict(zip(manifest["ids"], rows, strict=True))
    assert by_id["c0"] == [1.0, 2.0]
    assert by_id["c2"] == [3.0, 4.0]

    # chunks/ still holds a file for EVERY chunk, not just the embedded ones.
    chunk_files = sorted(p.name for p in (tmp_path / "chunks").glob("chunk_*.json"))
    assert chunk_files == ["chunk_0000.json", "chunk_0001.json", "chunk_0002.json"]


def test_fully_embedded_ids_cover_every_chunk_in_order(tmp_path: Path) -> None:
    space = _space([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    IndxWriter().write(space, tmp_path)

    manifest = json.loads((tmp_path / "embeddings" / "manifest.json").read_text())
    assert manifest["count"] == len(space.chunks) == 3
    assert manifest["ids"] == ["c0", "c1", "c2"]

    rows = _read_rows(tmp_path, manifest["dim"])
    assert rows == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
