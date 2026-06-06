"""JsonlStore — zero-dependency in-memory / no-DB vector store.

Brute-force cosine search. Self-contained (vectors travel inside the .indx archive), so a
space built with this store is fully air-gapped and portable (PRD US-3, NFR-PORT-1). Backs
the no-DB ``indx query`` path (OQ-3).
"""

from __future__ import annotations

import math

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit


class JsonlStore:
    name = "jsonl"

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        hits = [
            SearchHit(c, _cosine(vector, c.embedding))
            for c in self._chunks.values()
            if c.embedding is not None
        ]
        # Tie-break on chunk id so ordering is deterministic (NFR-DET-1).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):  # dim drift would yield a silently-wrong score; fail fast
        raise StageError("store", f"dimension mismatch: query has {len(a)}, stored has {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
