"""T1 — JsonlStore search contract: top-k, descending order, deterministic ties."""

from __future__ import annotations

import pytest

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.jsonl import JsonlStore


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)


def test_search_returns_topk_in_descending_order() -> None:
    store = JsonlStore()
    store.upsert([_chunk("a", [1, 0]), _chunk("b", [0, 1]), _chunk("c", [1, 1])])
    hits = store.search([1, 0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score


def test_search_dimension_mismatch_raises() -> None:
    # Dim drift must fail fast rather than return a silently-wrong score (NFR-DET-1).
    store = JsonlStore()
    store.upsert([_chunk("a", [1, 0])])
    with pytest.raises(StageError):
        store.search([1, 0, 0], k=5)


def test_search_equal_length_path_unchanged() -> None:
    store = JsonlStore()
    store.upsert([_chunk("a", [1, 0]), _chunk("b", [0, 1])])
    hits = store.search([1, 0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)


def test_delete_removes_chunk() -> None:
    store = JsonlStore()
    store.upsert([_chunk("a", [1, 0])])
    store.delete(["a"])
    assert store.search([1, 0], k=5) == []


def test_protocol_conformance() -> None:
    from indx.store.base import VectorStore

    assert isinstance(JsonlStore(), VectorStore)
