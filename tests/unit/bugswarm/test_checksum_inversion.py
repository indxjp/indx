"""Regression tests for checksum inversion / ghost-entry tamper detection in reader.py.

Finding: when an attacker deletes a member from the zip body but keeps its entry in
checksums.json, the reader must raise ArchiveError instead of silently skipping the member.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import indx.archive.reader as reader
from indx.archive import format as fmt
from indx.errors import ArchiveError

_MANIFEST_CONTENT = json.dumps({"schema_version": fmt.SCHEMA_VERSION}, sort_keys=True) + "\n"


def _write_archive(dest: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_ghost_entry_raises_archive_error(tmp_path: Path) -> None:
    """checksums.json lists documents.jsonl but the zip body omits it — must be detected.

    The manifest is given the correct checksum so that check passes; only the ghost entry
    for documents.jsonl (present in checksums.json but absent from the zip) should trigger
    the ArchiveError.
    """
    dest = tmp_path / "tampered.indx"
    # Compute the real manifest checksum so it passes validation.
    manifest_hash = hashlib.sha256(_MANIFEST_CONTENT.encode("utf-8")).hexdigest()
    # Build checksums.json that references documents.jsonl with a plausible hash,
    # but do NOT include documents.jsonl in the archive body.
    checksums = json.dumps(
        {
            "members": {
                fmt.MANIFEST: manifest_hash,
                fmt.DOCUMENTS: "a" * 64,  # fake sha256 hex digest for the ghost member
            }
        }
    )
    _write_archive(
        dest,
        {
            fmt.MANIFEST: _MANIFEST_CONTENT,
            fmt.CHECKSUMS: checksums,
            # fmt.DOCUMENTS is intentionally absent to simulate tampered archive
        },
    )
    with pytest.raises(ArchiveError, match="tampered|absent"):
        reader.read_archive(dest)
