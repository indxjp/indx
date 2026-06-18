"""IndxWriter — the default writer.

Writes the **expanded on-disk layout** (technical-spec §5.5): an output *directory* receives
the sealed ``<name>.indx`` archive plus ``index.json`` (the knowledge graph), per-chunk files
under ``chunks/``, and the vector matrix + manifest under ``embeddings/``. As a convenience,
if ``dest`` ends in ``.indx`` only the sealed archive is written (single-file mode).
"""

from __future__ import annotations

import array
import json
import sys
from pathlib import Path

from indx.archive import format as fmt
from indx.archive.writer import projected_documents, write_archive
from indx.core.knowledge_space import KnowledgeSpace
from indx.errors import IndxError


class IndxWriter:
    name = "indx"

    def write(self, space: KnowledgeSpace, dest: Path, *, name: str = "handbook") -> None:
        dest = Path(dest)
        if dest.suffix == ".indx":  # single-file mode
            write_archive(space, dest)
            return

        dest.mkdir(parents=True, exist_ok=True)
        write_archive(space, dest / f"{name}.indx")
        _write_index_json(space, dest / "index.json")
        _write_chunks(space, dest / "chunks")
        _write_embeddings(space, dest / "embeddings")


def _stats(space: KnowledgeSpace) -> dict[str, object]:
    from collections import Counter

    return {
        "documents": len(space.documents_),
        "chunks": len(space.chunks),
        "relations": len(space.relations),
        "embeddings": sum(1 for c in space.chunks if c.embedding is not None),
        "embed_dim": space.manifest.embedding_dim,
        "types": dict(Counter(d.doc_type or "unknown" for d in space.documents_)),
    }


def _write_index_json(space: KnowledgeSpace, path: Path) -> None:
    # Embeddings are never inlined into index.json — they live under embeddings/ (§6).
    index = {
        "version": fmt.SCHEMA_VERSION,
        "root": space.manifest.source_root,
        "metadata": space.manifest.model_dump(),
        "stats": _stats(space),
        # Same serialize-time projection the .indx archive uses, so index.json and the archive's
        # documents member agree: chunk_ids / references / referenced_by are backfilled rather than
        # shipped empty from the raw in-memory docs no stage fills (issue #17 bug 3 / #14 N3).
        "documents": list(projected_documents(space)),
        "chunks": [c.model_dump(exclude={"embedding"}) for c in space.chunks],
        "relations": [r.model_dump(mode="json") for r in space.relations],
    }
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_chunks(space: KnowledgeSpace, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("chunk_*.json"):
        stale.unlink()
    for i, chunk in enumerate(space.chunks):
        body = chunk.model_dump(exclude={"embedding"})
        (dest / f"chunk_{i:04d}.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _write_embeddings(space: KnowledgeSpace, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # vectors.f32 packs only embedded chunks (unembedded ones contribute no row), so the row
    # index is NOT the chunk index. Record the contributing chunk ids in row order alongside the
    # matrix so a reader can recover the row→chunk-id mapping (the bare matrix cannot, once any
    # chunk is unembedded). `ids` is row-aligned: len(ids) == count == number of rows.
    vectors: list[list[float]] = []
    ids: list[str] = []
    for chunk in space.chunks:
        if chunk.embedding is not None:
            vectors.append(chunk.embedding)
            ids.append(chunk.id)
    dim = len(vectors[0]) if vectors else (space.manifest.embedding_dim or 0)
    if vectors and any(len(v) != dim for v in vectors):
        raise IndxError("ragged embedding dimensions: all chunk embeddings must share one width")
    flat = array.array("f", [v for vec in vectors for v in vec])
    if sys.byteorder == "big":  # vectors.f32 is little-endian on disk
        flat.byteswap()
    (dest / "vectors.f32").write_bytes(flat.tobytes())
    manifest = {
        "model": space.manifest.embedding_model,
        "dim": dim,
        "count": len(vectors),
        "ids": ids,
        "backend": space.manifest.components.get("store"),
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
