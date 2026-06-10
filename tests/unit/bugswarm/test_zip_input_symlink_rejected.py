"""Regression: a symlink member in a build-input ``.zip`` is rejected, not extracted.

The module docstring promises symlink members never escape the extraction tree. This proves
the promise is backed by a real ``S_IFLNK`` check (raising :class:`ZipInputError`) rather than
only the implicit "we never honor symlinks" behavior, and that no symlink/file is materialized.
Fully offline — builds the archive in-process.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from indx.utils.zip_input import ZipInputError, extract_zip


def _live_temp_dirs() -> list[Path]:
    return list(Path("/tmp").glob("indx-zip-*")) if Path("/tmp").is_dir() else []


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    evil = tmp_path / "link.zip"
    # 0o120000 == S_IFLNK; the entry's payload is the link target string.
    link_info = zipfile.ZipInfo("evil_link")
    link_info.external_attr = (0o120777) << 16
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr(link_info, "/etc/passwd")

    before = set(_live_temp_dirs())
    with pytest.raises(ZipInputError, match="symlink"):
        extract_zip(evil)
    # The partial extraction dir is removed on failure (no leak).
    assert set(_live_temp_dirs()) == before
