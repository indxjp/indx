"""LangChainWriter — emit indx chunks as LangChain ``Document`` objects as JSONL.

Each indx :class:`~indx.core.chunk.Chunk` becomes a
``langchain_core.documents.Document`` (``page_content`` = chunk text, with provenance and
enrichment folded into ``metadata``) and the documents are serialized to newline-delimited
JSON at ``dest/<name>.jsonl``. The ``langchain-core`` package is an optional extra, so it is
imported lazily inside :meth:`LangChainWriter.write` and gated by
:func:`~indx.utils.lazy.require_extra` in ``__init__``; importing this module is always safe
(file-architecture §5, coding-standards §6.3). Vendor ``Document`` objects are constructed
and serialized entirely inside :meth:`write`; they never enter a core model
(coding-standards §6.2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from indx.core.knowledge_space import KnowledgeSpace
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from langchain_core.documents import (  # type: ignore[import-not-found]  # optional extra: langchain
        Document,
    )


class LangChainWriter:
    """Writer that converts a :class:`KnowledgeSpace` into LangChain ``Document`` JSONL.

    Mirrors the ``jsonl`` writer's destination conventions: ``dest`` is a directory and a
    single ``<name>.jsonl`` file is written into it, one ``Document`` per line. Each chunk's
    text becomes ``page_content`` and its provenance (source path, lineage), neighbor links,
    and the parent document's enrichment (doc type, topics, tags) are folded into
    JSON-primitive ``metadata`` — downstream vector stores reject nested objects.

    Attributes:
        name: Registry name for this writer.
    """

    name = "langchain"

    def __init__(self) -> None:
        """Construct the writer, verifying the optional ``langchain`` extra is present.

        Raises:
            MissingExtraError: If the ``langchain`` extra is not installed.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("writer", "langchain", "langchain", "langchain_core")

    def write(self, space: KnowledgeSpace, dest: Path, *, name: str = "handbook") -> None:
        """Serialize the space's chunks to ``dest/<name>.jsonl`` as LangChain documents.

        Args:
            space: The fully-populated knowledge space to serialize.
            dest: Destination directory; created if it does not exist.
            name: Base filename (without extension) for the JSONL output.

        Each serialized line carries LangChain's native ``Document`` shape from
        ``model_dump()`` -- ``id``, ``page_content``, ``metadata``, and LangChain's
        ``type`` discriminator (value ``"Document"``) -- kept intact so the JSONL loads
        straight back into LangChain.
        """
        from langchain_core.documents import Document  # optional extra: langchain

        documents: list[Document] = []
        for chunk in space.chunks:
            parent = space.document(chunk.doc_id)
            # Keep metadata JSON-primitive: downstream vector stores reject nested objects.
            documents.append(
                Document(
                    id=chunk.id,
                    page_content=chunk.text,
                    metadata={
                        "doc_id": chunk.doc_id,
                        "position": chunk.position,
                        "prev_id": chunk.prev_id,
                        "next_id": chunk.next_id,
                        "source": parent.path if parent is not None else None,
                        "lineage": list(parent.lineage) if parent is not None else [],
                        "doc_type": parent.doc_type if parent is not None else None,
                        "topics": list(parent.topics) if parent is not None else [],
                        "tags": list(parent.tags) if parent is not None else [],
                    },
                )
            )

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        _dump(dest / f"{name}.jsonl", (document.model_dump() for document in documents))


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as deterministic newline-delimited JSON (sorted keys)."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
