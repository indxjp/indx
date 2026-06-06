"""FakeStore — a trivial VectorStore that satisfies the protocol for discovery tests."""

from __future__ import annotations

from indx.core.chunk import Chunk
from indx.store.base import SearchHit


class FakeStore:
    name = "fake"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def upsert(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        return [SearchHit(c, 1.0) for c in self._chunks[:k]]

    def delete(self, chunk_ids: list[str]) -> None:
        self._chunks = [c for c in self._chunks if c.id not in set(chunk_ids)]
