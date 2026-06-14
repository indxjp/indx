"""Shared CLI helpers: format→writer mapping, space loading, Rich console."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console

from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.errors import ArchiveError

console = Console()

# `--format` / `[output] format` value -> writer slot name.
_WRITER_NAMES = {
    ".indx": "indx",
    "indx": "indx",
    "jsonl": "jsonl",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
}


def writer_name(fmt: str) -> str:
    """Map a documented output format to its writer slot name."""
    return _WRITER_NAMES.get(fmt, fmt)


def resolve_archive(path: Path) -> Path:
    """Resolve a CLI path to a ``.indx`` file.

    Accepts either the archive itself or an output directory containing one (``inspect`` /
    ``query`` take either form, per the docsite). Raises :class:`ArchiveError` if none found.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        preferred = path / "handbook.indx"
        if preferred.is_file():
            return preferred
        archives = sorted(path.glob("*.indx"))
        if archives:
            return archives[0]
    raise ArchiveError(f"no .indx archive found at: {path}")


def load_space(path: Path, *, members: Iterable[str] | None = None) -> KnowledgeSpace:
    """Load a :class:`KnowledgeSpace` from a CLI path, for both ``inspect`` and ``query``.

    Accepts the two on-disk shapes ``--out`` can produce (per the docsite, both ``inspect``
    and ``query`` "take a .indx archive or output directory"):

    * a ``.indx`` file — or a directory containing one — is read via :func:`read_archive`;
    * a directory holding the ``jsonl`` format (``manifest.json`` + ``documents.jsonl`` +
      ``chunks.jsonl`` + ``relations.jsonl``) is reconstructed shard by shard. ``chunks.jsonl``
      carries the embeddings written at build time, so :meth:`KnowledgeSpace.search` works.

    ``members`` is forwarded to :func:`read_archive` for the ``.indx`` path so ``inspect --part``
    can read just one member (Feature 1, selective archive load); it defaults to ``None`` (all),
    keeping existing callers unchanged. The jsonl-directory shape is plain files, so the hint is
    ignored there (the whole space loads and the caller slices it).

    Raises :class:`ArchiveError` (with the same actionable style as :func:`resolve_archive`)
    when ``path`` is neither.
    """
    from indx.archive import read_archive

    path = Path(path)
    if path.is_file() or (path.is_dir() and _has_indx(path)):
        return read_archive(resolve_archive(path), members=members)
    if path.is_dir() and (path / "manifest.json").is_file():
        return _load_jsonl_space(path)
    raise ArchiveError(f"no .indx archive or jsonl output directory found at: {path}")


def _has_indx(path: Path) -> bool:
    return (path / "handbook.indx").is_file() or bool(next(iter(path.glob("*.indx")), None))


def resolve_target(space: Path | None) -> KnowledgeSpace:
    """Load the space a command should act on: explicit path wins, else the home space (F4).

    * ``space`` given  -> :func:`load_space` (a ``.indx`` file or output dir, unchanged).
    * ``space`` None   -> :func:`indx.home.open_home` (load-or-init ``$INDX_HOME/space.indx``).

    Used by ``query`` / ``ask`` / ``add`` / ``rm`` / ``update`` / ``inspect`` so a bare
    ``indx query "..."`` / ``indx add <file>`` targets the home DB.
    """
    if space is not None:
        return load_space(space)
    from indx.home import open_home

    return open_home()


def resolve_target_path(space: Path | None) -> Path | None:
    """The archive path a mutating command should reseal to (``None`` => home) (F4).

    With an explicit ``space`` this resolves the ``.indx`` archive (a missing path raises
    :class:`ArchiveError`, exit 4). ``None`` signals the caller to reseal via
    :func:`indx.home.save_home`.
    """
    if space is None:
        return None
    return resolve_archive(space)


def parent_only(space: KnowledgeSpace) -> KnowledgeSpace:
    """Return a view that does NOT federate over children (Feature 2, ``--no-children``).

    A shallow copy whose ``manifest.children`` is empty, so ``stats``/``search`` operate on the
    parent's own rows. The source binding is preserved (harmless) and no child file is touched.
    """
    view = space.model_copy(update={"manifest": space.manifest.model_copy(update={"children": []})})
    if space._source_path is not None:
        view._bind_source(space._source_path)
    return view


def _load_jsonl_space(path: Path) -> KnowledgeSpace:
    """Reconstruct a KnowledgeSpace from the ``jsonl`` writer's on-disk shards."""
    from indx.core.chunk import Chunk
    from indx.core.document import Document
    from indx.core.relation import Relation

    manifest = Manifest.model_validate(json.loads((path / "manifest.json").read_text("utf-8")))
    space = KnowledgeSpace(
        manifest=manifest,
        documents=[Document.model_validate(r) for r in _read_jsonl(path / "documents.jsonl")],
        chunks=[Chunk.model_validate(r) for r in _read_jsonl(path / "chunks.jsonl")],
        relations=[Relation.model_validate(r) for r in _read_jsonl(path / "relations.jsonl")],
    )
    # Bind the source dir so relative child refs (Feature 2) resolve against it. Compose targets
    # .indx archives, but a jsonl dir that carries a children manifest can still federate on read.
    space._bind_source(path)
    return space


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
