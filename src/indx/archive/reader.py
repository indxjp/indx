"""Read a .indx archive back into a KnowledgeSpace, checking schema compatibility."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from indx.archive import format as fmt
from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.core.relation import Relation
from indx.errors import ArchiveError

_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Decompression budget per member. The .indx format is portable/shareable, so an archive may
# come from an untrusted source; a tiny compressed member can otherwise inflate to gigabytes and
# exhaust memory before any validation runs (zip-bomb). 256 MiB is far above any realistic member
# while still capping a hostile payload. Enforced both against the declared uncompressed size and
# during the actual read, since a crafted central directory can understate ``file_size``.
MAX_MEMBER_BYTES = 256 * 1024 * 1024


def _read_member(zf: zipfile.ZipFile, name: str, src: Path) -> bytes:
    """Read one archive member with a hard byte budget (zip-bomb defence)."""
    declared = zf.getinfo(name).file_size
    if declared > MAX_MEMBER_BYTES:
        raise ArchiveError(
            f"member {name!r} too large in {src}: {declared} bytes exceeds limit {MAX_MEMBER_BYTES}"
        )
    with zf.open(name) as member:
        # Read one byte past the budget so an understated file_size still trips the ceiling.
        data = member.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ArchiveError(
            f"member {name!r} too large in {src}: decompressed size exceeds "
            f"limit {MAX_MEMBER_BYTES}"
        )
    return data


def read_archive(src: Path) -> KnowledgeSpace:
    src = Path(src)
    if not zipfile.is_zipfile(src):
        raise ArchiveError(f"not a .indx archive (not a zip): {src}")
    with zipfile.ZipFile(src) as zf:
        names = set(zf.namelist())
        if fmt.MANIFEST not in names:
            raise ArchiveError(f"archive missing {fmt.MANIFEST}: {src}")
        try:
            manifest = Manifest.model_validate(json.loads(_read_member(zf, fmt.MANIFEST, src)))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise ArchiveError(f"corrupt {fmt.MANIFEST} in {src}: {exc}") from exc
        if manifest.schema_version != fmt.SCHEMA_VERSION:
            raise ArchiveError(
                f"unsupported .indx schema_version {manifest.schema_version!r} "
                f"(reader supports {fmt.SCHEMA_VERSION!r})"
            )
        _verify_checksums(zf, names, src)
        return KnowledgeSpace(
            manifest=manifest,
            documents=_load(zf, fmt.DOCUMENTS, Document, src),
            chunks=_load(zf, fmt.CHUNKS, Chunk, src),
            relations=_load(zf, fmt.RELATIONS, Relation, src),
        )


def _verify_checksums(zf: zipfile.ZipFile, names: set[str], src: Path) -> None:
    """Validate per-member SHA-256 digests (indx-archive.md "Loading", step 2).

    Archives written before checksums existed simply omit ``checksums.json``; for forward
    tolerance their integrity check is skipped rather than treated as a failure. When the
    member is present, every recorded digest must match the member's current bytes.
    """
    if fmt.CHECKSUMS not in names:
        return
    try:
        record = json.loads(_read_member(zf, fmt.CHECKSUMS, src).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"corrupt {fmt.CHECKSUMS} in {src}: {exc}") from exc
    if not isinstance(record, dict):
        raise ArchiveError(f"corrupt {fmt.CHECKSUMS} in {src}: expected an object")
    expected = record.get("members", {})
    if not isinstance(expected, dict):
        raise ArchiveError(f"corrupt {fmt.CHECKSUMS} in {src}: 'members' must be an object")
    for name, want in expected.items():
        if name not in names:
            raise ArchiveError(f"archive missing checksummed member {name!r}: {src}")
        try:
            got = fmt.checksum(_read_member(zf, name, src).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ArchiveError(f"corrupt {name} in {src}: {exc}") from exc
        if got != want:
            raise ArchiveError(
                f"checksum mismatch for {name!r} in {src}: expected {want}, got {got}"
            )


def _load(zf: zipfile.ZipFile, name: str, model: type[_ModelT], src: Path) -> list[_ModelT]:
    try:
        return [model.model_validate(r) for r in _rows(zf, name, src)]
    except ValidationError as exc:
        raise ArchiveError(f"corrupt {name} in {src}: {exc}") from exc


def _rows(zf: zipfile.ZipFile, name: str, src: Path) -> list[dict[str, Any]]:
    if name not in zf.namelist():
        return []
    try:
        text = _read_member(zf, name, src).decode("utf-8")
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"corrupt {name} in {src}: {exc}") from exc
