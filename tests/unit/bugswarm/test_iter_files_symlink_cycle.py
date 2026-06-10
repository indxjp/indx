"""Regression: ``iter_files(follow_symlinks=True)`` must not recurse on symlink cycles.

Previously the recursive directory walk followed symlinked directories without any
cycle detection (no visited resolved-path set), so a directory symlink cycle such as
``sub/loop -> <root>`` produced unbounded recursion and a ``RecursionError``. The walk
now tracks already-visited real paths and short-circuits re-entry.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from indx.utils.io import iter_files


def test_follow_symlinks_directory_cycle_terminates(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("hi", encoding="utf-8")
    # Create a directory symlink cycle: sub/loop -> tmp_path (the root).
    try:
        os.symlink(tmp_path, sub / "loop", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    files = list(iter_files(tmp_path, follow_symlinks=True))

    # Must terminate (no RecursionError) and yield the single real file exactly once.
    assert [p.name for p in files] == ["a.txt"]


def test_default_does_not_follow_symlinks(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("hi", encoding="utf-8")
    try:
        os.symlink(tmp_path, sub / "loop", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    files = list(iter_files(tmp_path))

    assert [p.name for p in files] == ["a.txt"]
