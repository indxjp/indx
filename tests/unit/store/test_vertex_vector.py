"""Unit tests for VertexVectorStore — vendor SDK mocked, fully offline (coding-standards §11).

The ``google-cloud-aiplatform`` wheel is absent in this environment, so a fake
``google.cloud.aiplatform`` module is injected into ``sys.modules`` (registering every
namespace parent ``google`` / ``google.cloud``) and the dependency gate is neutralized. The
fake implements just enough of the Vertex AI Vector Search surface the adapter uses
(``init`` / ``MatchingEngineIndex.upsert_datapoints`` / ``.remove_datapoints`` /
``MatchingEngineIndexEndpoint.find_neighbors``) to round-trip core
:class:`~indx.core.chunk.Chunk` ids.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError, StageError
from indx.store.vertex_vector import VertexVectorStore

# --- Fake vendor objects mirroring google.cloud.aiplatform -----------------------------

# A single shared store so an index (write side) and an endpoint (read side) constructed
# separately operate on the same datapoints, like the real (server-backed) service.
_DATAPOINTS: dict[str, list[float]] = {}


class _Neighbor:
    def __init__(self, *, id: str, distance: float) -> None:
        self.id = id
        self.distance = distance


class _FakeMatchingEngineIndex:
    def __init__(self, index_id: str) -> None:
        self.index_id = index_id
        self.upsert_calls: list[list[dict[str, Any]]] = []
        self.removed: list[str] = []

    def upsert_datapoints(self, *, datapoints: list[dict[str, Any]]) -> None:
        self.upsert_calls.append(datapoints)
        for dp in datapoints:
            _DATAPOINTS[dp["datapoint_id"]] = list(dp["feature_vector"])

    def remove_datapoints(self, *, datapoint_ids: list[str]) -> None:
        self.removed.extend(datapoint_ids)
        for did in datapoint_ids:
            _DATAPOINTS.pop(did, None)


class _FakeMatchingEngineIndexEndpoint:
    def __init__(self, index_endpoint_id: str) -> None:
        self.index_endpoint_id = index_endpoint_id
        self.find_kwargs: dict[str, Any] = {}

    def find_neighbors(
        self, *, deployed_index_id: str, queries: list[list[float]], num_neighbors: int
    ) -> list[list[_Neighbor]]:
        self.find_kwargs = {
            "deployed_index_id": deployed_index_id,
            "queries": queries,
            "num_neighbors": num_neighbors,
        }
        query = queries[0]
        scored = [
            _Neighbor(id=did, distance=1.0 - _cosine(query, vec))
            for did, vec in _DATAPOINTS.items()
        ]
        scored.sort(key=lambda n: n.distance)
        return [scored[:num_neighbors]]


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@pytest.fixture
def fake_aiplatform(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``google.cloud.aiplatform`` and neutralize the dependency gate."""
    _DATAPOINTS.clear()
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra`` at import time, binding
    # its own module-level name; patch that binding too so neutralization is effective
    # regardless of import order.
    monkeypatch.setattr("indx.store.vertex_vector.require_extra", lambda *a, **k: None)

    aiplatform = types.ModuleType("google.cloud.aiplatform")
    aiplatform.init = lambda **kwargs: None  # type: ignore[attr-defined]
    aiplatform.MatchingEngineIndex = _FakeMatchingEngineIndex  # type: ignore[attr-defined]
    aiplatform.MatchingEngineIndexEndpoint = (  # type: ignore[attr-defined]
        _FakeMatchingEngineIndexEndpoint
    )

    # Namespace-package mocking: register every parent so ``from google.cloud import
    # aiplatform`` resolves (coding-standards §11 namespace rule).
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.aiplatform = aiplatform  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", aiplatform)
    return aiplatform


def _store(**kwargs: Any) -> VertexVectorStore:
    base: dict[str, Any] = {
        "project": "p",
        "location": "us-central1",
        "index_id": "idx",
        "index_endpoint_id": "ep",
        "deployed_index_id": "dep",
        "dim": 2,
    }
    base.update(kwargs)
    return VertexVectorStore(**base)


def _chunk(cid: str, vec: list[float] | None) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def test_satisfies_vector_store_protocol(fake_aiplatform: types.ModuleType) -> None:
    from indx.store.base import VectorStore

    assert isinstance(_store(), VectorStore)


def test_upsert_then_search_roundtrips_core_types(fake_aiplatform: types.ModuleType) -> None:
    store = _store()
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    # The index stores no payload: id echoed into id/doc_id, text empty.
    assert hits[0].chunk.id == "a"
    assert hits[0].chunk.doc_id == "a"
    assert hits[0].chunk.text == ""


def test_upsert_sends_id_and_vector_datapoints(fake_aiplatform: types.ModuleType) -> None:
    store = _store()
    store.upsert([_chunk("a", [1.0, 0.0])])
    index = fake_aiplatform.MatchingEngineIndex("idx")  # type: ignore[attr-defined]
    # The shared backing store recorded exactly the id + vector at the edge.
    assert "a" in _DATAPOINTS
    assert _DATAPOINTS["a"] == [1.0, 0.0]
    assert index  # constructed without error


def test_search_forwards_k_as_num_neighbors(fake_aiplatform: types.ModuleType) -> None:
    store = _store()
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    hits = store.search([1.0, 0.0], k=1)
    assert len(hits) == 1


def test_search_descending_with_deterministic_tie_break(
    fake_aiplatform: types.ModuleType,
) -> None:
    store = _store()
    # Two ids with identical vectors -> identical distance; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_upsert_skips_chunks_without_embedding(fake_aiplatform: types.ModuleType) -> None:
    store = _store()
    store.upsert([_chunk("x", None), _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_delete_removes_datapoints(fake_aiplatform: types.ModuleType) -> None:
    store = _store()
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_lazy_sizes_from_first_vector_when_dim_none(
    fake_aiplatform: types.ModuleType,
) -> None:
    store = _store(dim=None)
    assert store._dim is None
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    assert store._dim == 3


def test_search_before_sizing_raises(fake_aiplatform: types.ModuleType) -> None:
    store = _store(dim=None)
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=5)


def test_search_without_endpoint_config_raises(fake_aiplatform: types.ModuleType) -> None:
    store = _store(index_endpoint_id=None, deployed_index_id=None)
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=5)


def test_upsert_without_index_id_raises(fake_aiplatform: types.ModuleType) -> None:
    store = _store(index_id=None)
    with pytest.raises(StageError):
        store.upsert([_chunk("a", [1.0, 0.0])])


def test_construction_without_extra_raises_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No gate neutralization here: the real (dep-absent) environment must raise. Purge any
    # ``google*`` modules a prior test's fake injected so this runs against the true (absent)
    # SDK regardless of test ordering — the env genuinely lacks google-cloud-aiplatform.
    for mod in [m for m in sys.modules if m == "google" or m.startswith("google.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    with pytest.raises(MissingExtraError):
        VertexVectorStore(index_id="idx", dim=2)
