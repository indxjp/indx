"""Regression: write_archive must be atomic — a mid-write I/O error must not
truncate dest or destroy a previously valid archive at that path."""

from __future__ import annotations

import zipfile

import pytest

from indx.archive import writer
from indx.core.knowledge_space import KnowledgeSpace


def test_failed_write_preserves_previous_archive(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "s.indx"
    previous = b"PREVIOUS GOOD ARCHIVE"
    dest.write_bytes(previous)

    # Simulate an I/O error during the compressed write phase (e.g. ENOSPC).
    real_writestr = zipfile.ZipFile.writestr

    def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        real_writestr(self, *args, **kwargs)
        raise OSError("no space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", boom)

    with pytest.raises(OSError, match="no space left"):
        writer.write_archive(KnowledgeSpace(), dest)

    # The old archive is untouched and no partial temp file leaks into the dir.
    assert dest.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [dest]


def test_successful_write_replaces_atomically(tmp_path) -> None:
    dest = tmp_path / "s.indx"
    writer.write_archive(KnowledgeSpace(), dest)

    assert zipfile.is_zipfile(dest)
    # No temp artifact left behind on success.
    assert list(tmp_path.iterdir()) == [dest]
