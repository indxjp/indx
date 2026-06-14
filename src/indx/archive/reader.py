"""Read a .indx archive back into a KnowledgeSpace, checking schema compatibility."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final, TypeVar

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

# The loadable content members (Feature 1: selective archive load). The manifest is always read
# for the version/integrity guard, so it is deliberately NOT a selectable "member" here. Each
# entry maps a public member key to its zip member filename and the model used to validate rows.
_LOADABLE: Final[dict[str, tuple[str, type[BaseModel]]]] = {
    "documents": (fmt.DOCUMENTS, Document),
    "chunks": (fmt.CHUNKS, Chunk),
    "relations": (fmt.RELATIONS, Relation),
}


def _resolve_members(members: Iterable[str] | None) -> set[str]:
    """Validate a requested member subset (Feature 1).

    ``None`` → all content members. A sequence is validated against :data:`_LOADABLE`; an unknown
    name (including ``"manifest"``, which is always read and never a loadable content member) raises
    :class:`ArchiveError` (exit 4) naming the offender and the valid set.
    """
    if members is None:
        return set(_LOADABLE)
    wanted = set(members)
    unknown = sorted(wanted - set(_LOADABLE))
    if unknown:
        raise ArchiveError(
            f"unknown archive member(s) {unknown}; valid members: {sorted(_LOADABLE)} "
            "(the manifest is always read and is not a selectable member)"
        )
    return wanted


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


def read_archive(src: Path, *, members: Iterable[str] | None = None) -> KnowledgeSpace:
    """Read a sealed ``.indx`` archive into a :class:`KnowledgeSpace`.

    ``members=None`` (the default) reads all content members and verifies every checksum —
    unchanged behavior. A subset (e.g. ``("documents", "chunks")``) reads and validates **only**
    those members; omitted members are never decompressed and their list comes back empty
    (Feature 1, selective archive load). The manifest is always read and its ``schema_version``
    always checked, and the zip-bomb + checksum guards still cover every member that *is* read.
    An unknown member name raises :class:`ArchiveError` (exit 4).
    """
    src = Path(src)
    wanted = _resolve_members(members)
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
        if manifest.schema_version not in fmt.SUPPORTED_SCHEMA_VERSIONS:
            raise ArchiveError(
                f"unsupported .indx schema_version {manifest.schema_version!r} "
                f"(reader supports {sorted(fmt.SUPPORTED_SCHEMA_VERSIONS)!r})"
            )
        # Verify only the members we are about to read (plus the always-read manifest), so a
        # subset load never validates — or even decompresses — an omitted member's bytes.
        verify = {fmt.MANIFEST} | {_LOADABLE[k][0] for k in wanted}
        _verify_checksums(zf, names, src, verify=verify)
        space = KnowledgeSpace(
            manifest=manifest,
            documents=(_load(zf, fmt.DOCUMENTS, Document, src) if "documents" in wanted else []),
            chunks=_load(zf, fmt.CHUNKS, Chunk, src) if "chunks" in wanted else [],
            relations=_load(zf, fmt.RELATIONS, Relation, src) if "relations" in wanted else [],
        )
        # Record where the archive came from so relative child refs (Feature 2) resolve against
        # the parent's directory. This is a private, non-serialized runtime attribute.
        space._bind_source(src)
        return space


def _verify_checksums(zf: zipfile.ZipFile, names: set[str], src: Path, *, verify: set[str]) -> None:
    """Validate per-member SHA-256 digests (indx-archive.md "Loading", step 2).

    Archives written before checksums existed simply omit ``checksums.json``; for forward
    tolerance their integrity check is skipped rather than treated as a failure. When the
    member is present, every recorded digest is checked **only for the members in ``verify``**
    (the ones actually being read this call, plus the always-read manifest) — a member that is
    not being read has its checksum leg skipped, since its bytes are never loaded (Feature 1).
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
        if name not in verify:
            continue
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
