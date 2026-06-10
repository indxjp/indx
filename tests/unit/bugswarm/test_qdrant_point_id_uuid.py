"""Regression: QdrantStore must map arbitrary chunk ids to valid Qdrant point ids.

Qdrant accepts only an unsigned integer or a UUID string as a point id. Chunk ids are
16-char truncated SHA-256 hex strings (``indx.utils.hashing.stable_hash``) which are
neither, so a real Qdrant server rejects ``upsert``/``delete``. The store must derive a
deterministic UUID for the point id and carry the original chunk id in the payload so the
chunk-id contract survives the round-trip. Fully offline: the vendor SDK is faked, but the
fake mirrors the real server by *validating* the point id is a UUID.
"""

from __future__ import annotations

import sys
import types
import uuid
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.store.qdrant import QdrantStore


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


# --- Fake vendor objects that enforce the real Qdrant point-id contract ---------------


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


class _StrictQdrantClient:
    """Like the real server: rejects any point id that is not a UUID (or unsigned int)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._points: dict[str, _PointStruct] = {}
        self.collections: set[str] = set()

    @staticmethod
    def _check_id(pid: Any) -> None:
        if isinstance(pid, int) and pid >= 0:
            return
        if isinstance(pid, str):
            try:
                uuid.UUID(pid)
            except ValueError as exc:
                raise ValueError(f"not a valid point ID: {pid!r}") from exc
            return
        raise ValueError(f"not a valid point ID: {pid!r}")

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, *, vectors_config: Any) -> None:
        self.collections.add(name)

    def upsert(self, collection_name: str, *, points: list[_PointStruct], wait: bool) -> None:
        for p in points:
            self._check_id(p.id)
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
            _ScoredPoint(id=p.id, score=1.0, payload=dict(p.payload), vector=list(p.vector))
            for p in self._points.values()
        ]
        return _QueryResponse(scored[:limit])

    def delete(self, collection_name: str, *, points_selector: _PointIdsList) -> None:
        for pid in points_selector.points:
            self._check_id(pid)
            self._points.pop(pid, None)


@pytest.fixture
def strict_qdrant(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.qdrant.require_extra", lambda *a, **k: None)
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = _StrictQdrantClient  # type: ignore[attr-defined]
    module.models = _FakeModels  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return module


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def test_upsert_uses_uuid_point_ids_and_roundtrips_chunk_id(
    strict_qdrant: types.ModuleType,
) -> None:
    store = QdrantStore(dim=2)
    # A 16-char truncated-sha256-style hex id: neither an integer nor a valid UUID.
    raw_id = "a1b2c3d4e5f60718"
    store.upsert([_chunk(raw_id, [1.0, 0.0])])

    client: _StrictQdrantClient = store._client  # type: ignore[assignment]
    # The stored point id is a valid UUID (would be rejected by a real server otherwise).
    (point_id,) = client._points.keys()
    assert _is_uuid(point_id)
    assert point_id != raw_id

    # The original chunk id survives the round-trip via the payload.
    hits = store.search([1.0, 0.0], k=1)
    assert hits[0].chunk.id == raw_id


def test_delete_addresses_the_derived_point_id(strict_qdrant: types.ModuleType) -> None:
    store = QdrantStore(dim=2)
    raw_id = "deadbeefcafe0123"
    store.upsert([_chunk(raw_id, [1.0, 0.0])])
    # delete() must convert the chunk id the same way; a strict server would 400 otherwise.
    store.delete([raw_id])
    assert store.search([1.0, 0.0], k=5) == []


def test_point_id_is_deterministic() -> None:
    from indx.store.qdrant import _point_id

    first = _point_id("a1b2c3d4e5f60718")
    assert first == _point_id("a1b2c3d4e5f60718")
    assert _is_uuid(first)
    assert first != _point_id("0000000000000000")
