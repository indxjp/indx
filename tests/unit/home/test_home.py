"""Feature 4 — ``src/indx/home.py`` (the persistent personal knowledge base).

All offline core stack; INDX_HOME is isolated to a tmp dir by the package conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import KnowledgeSpace
from indx.errors import ArchiveError
from indx.home import (
    home_dir,
    home_space_path,
    home_store_dir,
    open_home,
    reset_home,
    save_home,
)


def test_home_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``INDX_HOME`` set -> that path; unset -> ``~/.indx`` (via a monkeypatched Path.home)."""
    target = tmp_path / "custom-home"
    monkeypatch.setenv("INDX_HOME", str(target))
    assert home_dir() == target

    monkeypatch.delenv("INDX_HOME", raising=False)
    fake_home = tmp_path / "fakehome"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    assert home_dir() == fake_home / ".indx"


def test_open_home_initializes_empty() -> None:
    """A fresh home: ``open_home`` creates space.indx + store/, pinned to the offline stack."""
    space = open_home()
    assert home_space_path().is_file()
    assert home_store_dir().is_dir()
    assert space.manifest.components["embedder"] == "hash"
    assert space.manifest.components["llm"] == "none"
    assert space.stats.documents == 0


def test_open_home_loads_existing(tmp_path: Path) -> None:
    """Persisted mutations survive a fresh ``open_home`` (process-boundary surrogate)."""
    (tmp_path / "note.txt").write_text("# Note\n\npersisted body content.\n", encoding="utf-8")
    space = open_home()
    space.manifest.source_root = str(tmp_path)
    space.add("note.txt")
    save_home(space)

    reloaded = open_home()
    assert reloaded.stats.documents == 1
    assert any(d.path == "note.txt" for d in reloaded.documents())


def test_open_home_corrupt_raises_archiveerror() -> None:
    """A corrupt ``space.indx`` raises ArchiveError — it is NOT silently re-initialized."""
    open_home()  # create the dir + an initial archive
    home_space_path().write_bytes(b"not a zip archive at all")
    with pytest.raises(ArchiveError):
        open_home()


def test_reset_home_empties(tmp_path: Path) -> None:
    """``reset_home`` wipes the space + store, keeps the home dir; ``open_home`` re-inits empty."""
    (tmp_path / "x.txt").write_text("payload words", encoding="utf-8")
    space = open_home()
    space.manifest.source_root = str(tmp_path)
    space.add("x.txt")
    save_home(space)
    assert open_home().stats.documents == 1

    returned = reset_home()
    assert returned == home_dir()
    assert home_dir().is_dir()
    assert not home_space_path().is_file()
    assert open_home().stats.documents == 0


def test_save_home_round_trip_deterministic() -> None:
    """``save_home`` twice with identical content -> byte-identical space.indx."""
    space = open_home()
    save_home(space)
    first = home_space_path().read_bytes()
    save_home(space)
    second = home_space_path().read_bytes()
    assert first == second
    # And it round-trips back to a KnowledgeSpace.
    assert isinstance(KnowledgeSpace.load(str(home_space_path())), KnowledgeSpace)
