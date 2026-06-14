"""indx home — one persistent personal knowledge base under ~/.indx (override $INDX_HOME).

Zero heavy imports at module top (``pathlib``/``os`` only). All filesystem-touching helpers
resolve the home dir **lazily on each call** (never cache ``~/.indx`` at import) so a test that
sets ``INDX_HOME`` after import still wins and the real ``~/.indx`` is never touched under test.
"""

from __future__ import annotations

import os
from pathlib import Path

from indx.core.knowledge_space import KnowledgeSpace, Manifest

HOME_ENV = "INDX_HOME"
_DEFAULT_DIRNAME = ".indx"
_SPACE_NAME = "space.indx"
_STORE_DIRNAME = "store"

# The offline core stack recorded into a freshly-initialized home manifest, so the home space
# loads-or-serves with zero deps / no API key. Mirrors the registry builtin slot names.
_HOME_COMPONENTS: dict[str, str] = {
    "parser": "plaintext",
    "llm": "none",
    "vlm": "none",
    "embedder": "hash",
    "store": "jsonl",
}


def home_dir() -> Path:
    """The home directory: ``$INDX_HOME`` if set (and non-empty), else ``~/.indx``.

    Resolved fresh on every call (never import-time) so tests that set ``INDX_HOME`` win and
    the real ``~/.indx`` is never touched under test. The path is ``expanduser``'d but not
    created here — :func:`open_home` is the only writer.
    """
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / _DEFAULT_DIRNAME


def home_space_path() -> Path:
    """Path to the home space archive (``$INDX_HOME/space.indx``)."""
    return home_dir() / _SPACE_NAME


def home_store_dir() -> Path:
    """Path to the home vector-store dir (``$INDX_HOME/store/``)."""
    return home_dir() / _STORE_DIRNAME


def _empty_home_space() -> KnowledgeSpace:
    """A fresh, empty home space pinned to the offline core stack."""
    return KnowledgeSpace(
        manifest=Manifest(source_root=str(home_dir()), components=dict(_HOME_COMPONENTS)),
    )


def open_home() -> KnowledgeSpace:
    """Load the home space, initializing an empty one (pinned offline) on first use.

    Creates ``$INDX_HOME`` and ``$INDX_HOME/store/`` if absent. If ``space.indx`` exists it is
    read via :meth:`KnowledgeSpace.load`; otherwise an empty space is built, sealed once so the
    archive exists, and returned. Never touches the real ``~/.indx`` under test (callers set
    ``INDX_HOME``). Raises :class:`~indx.errors.ArchiveError` only if an existing ``space.indx``
    is unreadable/corrupt (it is *not* silently re-initialized — that would lose data).
    """
    root = home_dir()
    root.mkdir(parents=True, exist_ok=True)
    home_store_dir().mkdir(parents=True, exist_ok=True)
    archive = home_space_path()
    if archive.is_file():
        return KnowledgeSpace.load(str(archive))
    space = _empty_home_space()
    space.save(str(archive))
    return space


def save_home(space: KnowledgeSpace) -> None:
    """Seal ``space`` back to ``$INDX_HOME/space.indx`` (deterministic re-write)."""
    home_dir().mkdir(parents=True, exist_ok=True)
    space.save(str(home_space_path()))


def reset_home() -> Path:
    """Delete the home space + store contents, returning the (now-empty) home dir.

    Removes ``space.indx`` and the ``store/`` subtree but leaves the home dir itself in place
    so a subsequent :func:`open_home` re-initializes cleanly. Returns :func:`home_dir`. Safe to
    call when nothing exists yet (idempotent no-op).
    """
    import shutil

    archive = home_space_path()
    if archive.is_file():
        archive.unlink()
    store = home_store_dir()
    if store.is_dir():
        shutil.rmtree(store, ignore_errors=True)
    return home_dir()
