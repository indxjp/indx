"""Unit tests for QdrantStore — vendor SDK mocked, fully offline (coding-standards §11).

The ``qdrant_client`` wheel is absent in this environment, so a fake ``qdrant_client``
module is injected into ``sys.modules`` and the dependency gate is neutralized. The fake
client implements just enough of the Qdrant 1.x surface the adapter uses
(``collection_exists`` / ``create_collection`` / ``upsert`` / ``query_points`` / ``delete``)
to round-trip core :class:`~indx.core.chunk.Chunk` objects.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError
from indx.store.qdrant import QdrantStore

# --- Fake vendor objects mirroring qdrant_client.models / QdrantClient ----------------


class _PointStruct:
    def __init__(self, *, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _VectorParams:
    def __init__(self, *, size: int, distance: Any) -> None:
        self.size = size
        self.distance = distance


class _Distance:
    COSINE = "Cosine"


class _PointIdsList:
    def __init__(self, *, points: list[str]) -> None:
        self.points = points


class _ScoredPoint:
    def __init__(
        self, *, id: str, score: float, payload: dict[str, Any], vector: list[float]
    ) -> None:
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector


class _QueryResponse:
    def __init__(self, points: list[_ScoredPoint]) -> None:
        self.points = points


class _FakeModels:
    PointStruct = _PointStruct
    VectorParams = _VectorParams
    Distance = _Distance
    PointIdsList = _PointIdsList


class _FakeQdrantClient:
    """In-memory stand-in: stores points and answers a deterministic cosine search."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self._points: dict[str, _PointStruct] = {}
        self.collections: set[str] = set()
        self.upsert_calls: list[int] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, *, vectors_config: Any) -> None:
        self.collections.add(name)
        self.vectors_config = vectors_config

    def upsert(self, collection_name: str, *, points: list[_PointStruct], wait: bool) -> None:
        self.upsert_calls.append(len(points))
        for p in points:
            self._points[p.id] = p

    def query_points(
        self,
        collection_name: str,
        *,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> _QueryResponse:
        scored = [
            _ScoredPoint(
                id=p.id,
                score=_cosine(query, p.vector),
                payload=dict(p.payload),
                vector=list(p.vector),
            )
            for p in self._points.values()
        ]
        scored.sort(key=lambda s: -s.score)
        return _QueryResponse(scored[:limit])

    def delete(self, collection_name: str, *, points_selector: _PointIdsList) -> None:
        for pid in points_selector.points:
            self._points.pop(pid, None)


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``qdrant_client`` module and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra`` at import time, binding
    # its own module-level name; patch that binding too so neutralization is effective
    # regardless of import order.
    monkeypatch.setattr("indx.store.qdrant.require_extra", lambda *a, **k: None)
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    module.models = _FakeModels  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return module


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def test_satisfies_vector_store_protocol(fake_qdrant: types.ModuleType) -> None:
    from indx.store.base import VectorStore

    assert isinstance(QdrantStore(dim=2), VectorStore)


def test_construction_creates_sized_collection(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    client: _FakeQdrantClient = store._client  # type: ignore[assignment]
    assert "indx" in client.collections
    assert client.vectors_config.size == 2
    # Embedded (no-server) default: a path is passed, not a url.
    assert "path" in client.init_kwargs


def test_url_overrides_to_remote_server(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(url="http://localhost:6333", dim=2)
    client: _FakeQdrantClient = store._client  # type: ignore[assignment]
    assert client.init_kwargs.get("url") == "http://localhost:6333"
    assert "path" not in client.init_kwargs


def test_upsert_then_search_roundtrips_core_types(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert hits[0].chunk.doc_id == "d"
    assert hits[0].chunk.text == "a"
    assert hits[0].chunk.embedding == [1.0, 0.0]


def test_search_descending_with_deterministic_tie_break(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    # Two chunks with identical vectors -> identical score; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_upsert_skips_chunks_without_embedding(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    no_vec = Chunk(id="x", doc_id="d", position=0, text="x", embedding=None)
    store.upsert([no_vec, _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_upsert_batches_large_input(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    chunks = [_chunk(f"c{i:04d}", [1.0, float(i)]) for i in range(600)]
    store.upsert(chunks)
    client: _FakeQdrantClient = store._client  # type: ignore[assignment]
    # 600 points in batches of 256 -> 256 + 256 + 88.
    assert client.upsert_calls == [256, 256, 88]


def test_delete_removes_points(fake_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        QdrantStore(dim=2)
