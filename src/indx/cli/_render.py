"""Shared CLI helpers: format→writer mapping, space loading, Rich console."""

from __future__ import annotations

import json
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


def load_space(path: Path) -> KnowledgeSpace:
    """Load a :class:`KnowledgeSpace` from a CLI path, for both ``inspect`` and ``query``.

    Accepts the two on-disk shapes ``--out`` can produce (per the docsite, both ``inspect``
    and ``query`` "take a .indx archive or output directory"):

    * a ``.indx`` file — or a directory containing one — is read via :func:`read_archive`;
    * a directory holding the ``jsonl`` format (``manifest.json`` + ``documents.jsonl`` +
      ``chunks.jsonl`` + ``relations.jsonl``) is reconstructed shard by shard. ``chunks.jsonl``
      carries the embeddings written at build time, so :meth:`KnowledgeSpace.search` works.

    Raises :class:`ArchiveError` (with the same actionable style as :func:`resolve_archive`)
    when ``path`` is neither.
    """
    from indx.archive import read_archive

    path = Path(path)
    if path.is_file() or (path.is_dir() and _has_indx(path)):
        return read_archive(resolve_archive(path))
    if path.is_dir() and (path / "manifest.json").is_file():
        return _load_jsonl_space(path)
    raise ArchiveError(f"no .indx archive or jsonl output directory found at: {path}")


def _has_indx(path: Path) -> bool:
    return (path / "handbook.indx").is_file() or bool(next(iter(path.glob("*.indx")), None))


def _load_jsonl_space(path: Path) -> KnowledgeSpace:
    """Reconstruct a KnowledgeSpace from the ``jsonl`` writer's on-disk shards."""
    from indx.core.chunk import Chunk
    from indx.core.document import Document
    from indx.core.relation import Relation

    manifest = Manifest.model_validate(json.loads((path / "manifest.json").read_text("utf-8")))
    return KnowledgeSpace(
        manifest=manifest,
        documents=[Document.model_validate(r) for r in _read_jsonl(path / "documents.jsonl")],
        chunks=[Chunk.model_validate(r) for r in _read_jsonl(path / "chunks.jsonl")],
        relations=[Relation.model_validate(r) for r in _read_jsonl(path / "relations.jsonl")],
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
