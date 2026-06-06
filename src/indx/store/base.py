"""VectorStore protocol: upsert / search / delete over embedded chunks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from indx.core.chunk import Chunk
from indx.core.source import Source


class SearchHit(BaseModel):
    """A retrieval result: the matched chunk, its score, and resolved neighbor chunks.

    Pydantic v2 model (data-models.md §SearchHit, sdk.md §search). Store backends construct
    it positionally — ``SearchHit(chunk, score)`` — which the custom ``__init__`` below
    preserves alongside the keyword form ``SearchHit(chunk=..., score=...)``.
    """

    chunk: Chunk
    score: float
    neighbors: list[Chunk] = Field(default_factory=list)

    def __init__(
        self,
        chunk: Chunk | None = None,
        score: float | None = None,
        /,
        **data: Any,
    ) -> None:
        # Accept the long-standing positional store contract SearchHit(chunk, score) while
        # still allowing the keyword form Pydantic generates.
        if chunk is not None:
            data.setdefault("chunk", chunk)
        if score is not None:
            data.setdefault("score", score)
        super().__init__(**data)

    @property
    def source(self) -> Source | None:
        """Provenance of the matched chunk — shorthand for ``hit.chunk.source``."""
        return self.chunk.source


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-k hits in descending score order (the cross-store contract)."""
        ...

    def delete(self, chunk_ids: list[str]) -> None: ...
