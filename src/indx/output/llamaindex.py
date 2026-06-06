"""LlamaIndexWriter — emit LlamaIndex ``TextNode`` objects as JSONL. Requires the
``llamaindex`` extra.

Each indx :class:`~indx.core.chunk.Chunk` becomes a ``llama_index.core.schema.TextNode``
(``text`` + ``metadata``, with neighbor/source relationships) and the nodes are serialized
to newline-delimited JSON at ``dest/<name>.jsonl``. The vendor SDK is heavy/optional, so it
is imported lazily inside ``__init__`` and gated by
:func:`~indx.utils.lazy.require_extra`; importing this module is always safe
(file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from indx.core.knowledge_space import KnowledgeSpace
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from llama_index.core.schema import (  # type: ignore[import-not-found]  # optional extra: llamaindex
        TextNode,
    )


class LlamaIndexWriter:
    """Writer that converts a :class:`KnowledgeSpace` into LlamaIndex ``TextNode`` JSONL.

    Mirrors the ``jsonl`` writer's destination conventions: ``dest`` is a directory and a
    single ``<name>.jsonl`` file is written into it, one ``TextNode`` per line. Neighbor
    links (``prev_id`` / ``next_id``) and the owning document map onto LlamaIndex
    ``PREVIOUS`` / ``NEXT`` / ``SOURCE`` relationships. Vendor ``TextNode`` objects are
    constructed and serialized entirely inside :meth:`write`; they never enter a core model.

    Attributes:
        name: Registry name for this writer.
    """

    name = "llamaindex"

    def __init__(self) -> None:
        """Construct the writer, verifying the optional ``llamaindex`` extra is present.

        Raises:
            MissingExtraError: If the ``llamaindex`` extra is not installed.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("writer", "llamaindex", "llamaindex", "llama_index")

    def write(self, space: KnowledgeSpace, dest: Path, *, name: str = "handbook") -> None:
        """Serialize the space's chunks to ``dest/<name>.jsonl`` as LlamaIndex nodes.

        Args:
            space: The fully-populated knowledge space to serialize.
            dest: Destination directory; created if it does not exist.
            name: Base filename (without extension) for the JSONL output.
        """
        from llama_index.core.schema import (  # optional extra: llamaindex
            NodeRelationship,
            RelatedNodeInfo,
            TextNode,
        )

        nodes: list[TextNode] = []
        for chunk in space.chunks:
            document = space.document(chunk.doc_id)
            relationships: dict[NodeRelationship, RelatedNodeInfo] = {
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk.doc_id),
            }
            if chunk.prev_id is not None:
                relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=chunk.prev_id)
            if chunk.next_id is not None:
                relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=chunk.next_id)
            nodes.append(
                TextNode(
                    id_=chunk.id,
                    text=chunk.text,
                    embedding=chunk.embedding,
                    metadata={
                        "doc_id": chunk.doc_id,
                        "position": chunk.position,
                        "prev_id": chunk.prev_id,
                        "next_id": chunk.next_id,
                        "source": document.path if document is not None else None,
                    },
                    relationships=relationships,
                )
            )

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        _dump(dest / f"{name}.jsonl", (node.model_dump() for node in nodes))


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as deterministic newline-delimited JSON (sorted keys)."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
