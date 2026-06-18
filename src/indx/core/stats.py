"""SpaceStats — aggregate counts for a knowledge space.

Surfaced via :attr:`KnowledgeSpace.stats <indx.core.knowledge_space.KnowledgeSpace.stats>`,
serialized under the ``stats`` key of ``index.json``, and emitted by ``indx inspect --json``
(sdk.md §stats, index-json.md §stats).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpaceStats(BaseModel):
    """Aggregate statistics for a :class:`~indx.core.knowledge_space.KnowledgeSpace`."""

    documents: int
    chunks: int
    relations: int
    embeddings: int
    embed_dim: int | None = None
    types: dict[str, int] = Field(default_factory=dict)
    bytes_source: int = 0
    unindexed_documents: int = 0
    unindexed_paths: list[str] = Field(default_factory=list)
