"""Regression tests for src/indx/archive/reader.py hardening.

Two confirmed findings:
  * a structurally-valid-but-wrong-typed ``checksums.json`` (top-level not an object, or a
    non-object ``members`` value) must raise ``ArchiveError``, not a raw ``AttributeError``;
  * an oversized (zip-bomb) member must raise ``ArchiveError`` instead of being fully
    decompressed into memory.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import indx.archive.reader as reader
from indx.archive import format as fmt
from indx.errors import ArchiveError

_MANIFEST = json.dumps({"schema_version": fmt.SCHEMA_VERSION}, sort_keys=True) + "\n"


def _write_archive(dest: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_checksums_top_level_not_object_raises_archive_error(tmp_path: Path) -> None:
    dest = tmp_path / "bad.indx"
    # Top-level JSON is a list, so the old ``record.get(...)`` raised AttributeError.
    _write_archive(dest, {fmt.MANIFEST: _MANIFEST, fmt.CHECKSUMS: "[1, 2, 3]"})
    with pytest.raises(ArchiveError):
        reader.read_archive(dest)


def test_checksums_members_not_object_raises_archive_error(tmp_path: Path) -> None:
    dest = tmp_path / "bad.indx"
    # ``members`` is a list, so the old ``expected.items()`` raised AttributeError.
    _write_archive(
        dest, {fmt.MANIFEST: _MANIFEST, fmt.CHECKSUMS: json.dumps({"members": ["a", "b"]})}
    )
    with pytest.raises(ArchiveError):
        reader.read_archive(dest)


def test_oversized_member_raises_archive_error_without_full_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Shrink the ceiling so we don't need a real gigabyte payload; a highly compressible
    # member that inflates past the (tiny) budget must be rejected by ArchiveError.
    monkeypatch.setattr(reader, "MAX_MEMBER_BYTES", 1024)
    dest = tmp_path / "bomb.indx"
    bomb = "A" * (1024 * 1024)  # 1 MiB of one byte -> compresses to almost nothing
    _write_archive(dest, {fmt.MANIFEST: _MANIFEST, fmt.CHUNKS: bomb})
    with pytest.raises(ArchiveError):
        reader.read_archive(dest)
