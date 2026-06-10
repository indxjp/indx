"""Path / encoding helpers and safe directory walking."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from indx.utils.cache import CACHE_DIRNAME

# Names never worth indexing; keeps Walk deterministic and noise-free.
_IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules", CACHE_DIRNAME}
_IGNORE_FILES = {".DS_Store"}


def iter_files(root: Path, *, follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield files under ``root`` in a deterministic (sorted) order, skipping junk.

    Sorting matters: the walk order is part of the deterministic golden (testing-strategy §3.5).

    When ``follow_symlinks`` is True, symlinked directories are traversed, but a set of
    already-visited real paths is tracked so directory symlink cycles (e.g. ``a/loop -> ..``)
    cannot trigger unbounded recursion.
    """
    yield from _iter_files(Path(root), follow_symlinks=follow_symlinks, visited=set())


def _iter_files(root: Path, *, follow_symlinks: bool, visited: set[Path]) -> Iterator[Path]:
    real = root.resolve()
    if real in visited:
        return
    visited.add(real)
    entries = sorted(root.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_symlink() and not follow_symlinks:
            continue
        if entry.is_dir():
            if entry.name in _IGNORE_DIRS:
                continue
            yield from _iter_files(entry, follow_symlinks=follow_symlinks, visited=visited)
        elif entry.is_file() and entry.name not in _IGNORE_FILES:
            yield entry


def read_text(path: Path) -> str:
    """Read text tolerantly; never raise on a stray byte (robustness, see messy_real corpus)."""
    return Path(path).read_text(encoding="utf-8", errors="replace")
