"""Serialize a KnowledgeSpace into a .indx Zip archive.

Member order and JSON key order are fixed, and timestamps are zeroed, so the same space
produces byte-identical archives across runs/machines — the determinism guardrail the
double-run Docker test relies on (testing-strategy §4.4g).
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from indx.archive import format as fmt
from indx.core.knowledge_space import KnowledgeSpace
from indx.core.relation import RelationType

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)  # Zip epoch; keeps archives reproducible


def write_archive(space: KnowledgeSpace, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = space.manifest.model_copy(update={"schema_version": fmt.SCHEMA_VERSION})

    # Content members first; their SHA-256 digests are then sealed into a checksums member so a
    # reader can detect corruption/tampering on load (indx-archive.md "Integrity" / §11). The
    # checksums member is written last and is the one member it does not cover (no self-digest).
    content: list[tuple[str, str]] = [
        (fmt.MANIFEST, json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n"),
        (fmt.DOCUMENTS, _jsonl(projected_documents(space))),
        (fmt.CHUNKS, _jsonl(c.model_dump() for c in space.chunks)),
        (fmt.RELATIONS, _jsonl(r.model_dump(mode="json") for r in space.relations)),
    ]
    checksums = {
        "algo": fmt.CHECKSUM_ALGO,
        "members": {name: fmt.checksum(data) for name, data in content},
    }
    members: list[tuple[str, str]] = [
        *content,
        (fmt.CHECKSUMS, json.dumps(checksums, indent=2, sort_keys=True) + "\n"),
    ]
    # Write to a temp file in the same directory, then atomically rename it into place so an
    # I/O error mid-write (e.g. ENOSPC during compression/flush) never truncates dest or
    # destroys a previously valid archive at this path (mirrors utils/cache.py:put).
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        os.close(fd)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in members:  # deterministic member order
                info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, data)
        os.replace(tmp, dest)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def projected_documents(space: KnowledgeSpace) -> Iterable[dict[str, Any]]:
    """Serialize documents with their derived doc->chunk / link adjacency populated.

    ``Document.chunk_ids`` / ``references`` / ``referenced_by`` are the documented read surface
    (data-models.md §Document) but no pipeline stage fills them — every archive used to ship them
    empty (issue #14 N3). Rather than carry them as mutable in-memory state, they are computed
    here at the serialize boundary as a pure projection of the rows already in the space:

    * ``chunk_ids`` ← the ids of the chunks whose ``doc_id`` is this document, in stored order;
    * ``references`` / ``referenced_by`` ← ``REFERENCES`` relations grouped by ``src`` / ``dst``.

    The projection is a deterministic function of the (already-canonical) document/chunk/relation
    lists, so a fresh build and a "build then add" reseal still serialize byte-identically, and a
    reader gets the populated surface back after ``load()``.
    """
    chunk_ids_by_doc: dict[str, list[str]] = {}
    for chunk in space.chunks:
        chunk_ids_by_doc.setdefault(chunk.doc_id, []).append(chunk.id)
    refs_by_src: dict[str, list[dict[str, Any]]] = {}
    refs_by_dst: dict[str, list[dict[str, Any]]] = {}
    for rel in space.relations:
        if rel.type == RelationType.REFERENCES:
            dumped = rel.model_dump(mode="json")
            refs_by_src.setdefault(rel.src, []).append(dumped)
            refs_by_dst.setdefault(rel.dst, []).append(dumped)

    for doc in space.documents_:
        row = doc.model_dump()
        row["chunk_ids"] = chunk_ids_by_doc.get(doc.id, [])
        row["references"] = refs_by_src.get(doc.id, [])
        row["referenced_by"] = refs_by_dst.get(doc.id, [])
        yield row


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
