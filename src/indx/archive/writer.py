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
        (fmt.DOCUMENTS, _jsonl(d.model_dump() for d in space.documents_)),
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


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
