"""Unit tests for S3VectorsStore — boto3 mocked, fully offline (coding-standards §11).

The ``boto3`` wheel is absent in this environment, so a fake ``boto3`` module is injected into
``sys.modules`` and the dependency gate is neutralized. The fake ``s3vectors`` client implements
just enough of the S3 Vectors surface the adapter uses (``create_index`` / ``put_vectors``
/ ``query_vectors`` / ``delete_vectors``) to round-trip core :class:`~indx.core.chunk.Chunk`
objects, and RECORDS the dimension each index is sized to so lazy-sizing can be pinned to the
vector width (mirrors ``tests/unit/test_store_lazy_dim.py``).
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError, StageError
from indx.store.s3vectors import S3VectorsStore

# --- Fake boto3 ``s3vectors`` client --------------------------------------------------


class _FakeS3VectorsClient:
    """In-memory stand-in: stores vectors and answers a deterministic cosine search."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.create_dims: list[int] = []
        self.indexes: set[str] = set()
        self._vectors: dict[str, dict[str, Any]] = {}

    def create_index(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        dataType: str,
        dimension: int,
        distanceMetric: str,
    ) -> None:
        self.create_calls.append(
            {
                "vectorBucketName": vectorBucketName,
                "indexName": indexName,
                "dataType": dataType,
                "dimension": dimension,
                "distanceMetric": distanceMetric,
            }
        )
        self.create_dims.append(dimension)
        self.indexes.add(indexName)

    def put_vectors(
        self, *, vectorBucketName: str, indexName: str, vectors: list[dict[str, Any]]
    ) -> None:
        for v in vectors:
            self._vectors[v["key"]] = v

    def query_vectors(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        queryVector: dict[str, list[float]],
        topK: int,
        returnDistance: bool,
        returnMetadata: bool,
    ) -> dict[str, Any]:
        q = queryVector["float32"]
        scored = [
            {
                "key": v["key"],
                "distance": 1.0 - _cosine(q, v["data"]["float32"]),
                "metadata": dict(v["metadata"]),
            }
            for v in self._vectors.values()
        ]
        # Nearest first = smallest distance first.
        scored.sort(key=lambda r: r["distance"])
        return {"vectors": scored[:topK]}

    def delete_vectors(self, *, vectorBucketName: str, indexName: str, keys: list[str]) -> None:
        for key in keys:
            self._vectors.pop(key, None)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class _FakeBoto3:
    """Records ``client(service, **kwargs)`` calls and hands back the s3vectors stub."""

    def __init__(self) -> None:
        self.client_calls: list[tuple[str, dict[str, Any]]] = []
        self.s3vectors = _FakeS3VectorsClient()

    def client(self, service: str, **kwargs: Any) -> _FakeS3VectorsClient:
        self.client_calls.append((service, kwargs))
        if service != "s3vectors":
            raise AssertionError(f"unexpected service: {service}")
        return self.s3vectors


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> _FakeBoto3:
    """Inject a fake ``boto3`` module and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra`` at import time, binding its
    # own module-level name; patch that binding too so neutralization is effective regardless of
    # import order.
    monkeypatch.setattr("indx.store.s3vectors.require_extra", lambda *a, **k: None)
    fake = _FakeBoto3()
    module = types.ModuleType("boto3")
    module.client = fake.client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)
    return fake


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def _store(fake: _FakeBoto3, **kwargs: Any) -> S3VectorsStore:
    return S3VectorsStore(bucket="b", **kwargs)


def test_satisfies_vector_store_protocol(fake_boto3: _FakeBoto3) -> None:
    from indx.store.base import VectorStore

    assert isinstance(_store(fake_boto3, dim=2), VectorStore)


def test_construction_creates_s3vectors_client(fake_boto3: _FakeBoto3) -> None:
    _store(fake_boto3, dim=2, region="us-east-1")
    assert fake_boto3.client_calls == [("s3vectors", {"region_name": "us-east-1"})]


def test_upsert_then_search_roundtrips_core_types(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert hits[0].chunk.doc_id == "d"
    assert hits[0].chunk.text == "a"


def test_search_descending_with_deterministic_tie_break(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=2)
    # Two chunks with identical vectors -> identical score; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_score_is_one_minus_distance(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=1)
    # Identical vector -> cosine distance 0 -> score 1.0 (higher is better).
    assert hits[0].score == pytest.approx(1.0)


def test_upsert_skips_chunks_without_embedding(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=2)
    no_vec = Chunk(id="x", doc_id="d", position=0, text="x", embedding=None)
    store.upsert([no_vec, _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_delete_removes_vectors(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


# --- Lazy dim sizing (mirrors tests/unit/test_store_lazy_dim.py) ----------------------


def test_with_dim_sizes_index_on_first_upsert(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3, dim=1536)
    store.upsert([_chunk("a", [0.0] * 1536)])
    assert fake_boto3.s3vectors.create_dims == [1536]


def test_no_dim_defers_then_sizes_to_first_vector(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3)  # no dim
    assert fake_boto3.s3vectors.create_dims == []  # deferred — nothing sized yet
    store.upsert([_chunk("a", [0.0] * 1024)])
    # Width comes from the upserted vector, not a hard-coded default.
    assert fake_boto3.s3vectors.create_dims == [1024]


def test_search_before_sizing_raises_typed_error(fake_boto3: _FakeBoto3) -> None:
    store = _store(fake_boto3)  # no dim, nothing upserted
    with pytest.raises(StageError):
        store.search([0.0, 1.0])


# --- Config & version guards ----------------------------------------------------------


def test_missing_bucket_raises_stage_error_on_upsert(fake_boto3: _FakeBoto3) -> None:
    store = S3VectorsStore(dim=2)  # no bucket
    with pytest.raises(StageError):
        store.upsert([_chunk("a", [1.0, 0.0])])


def test_old_boto3_without_s3vectors_client_raises_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # boto3 < 1.40 raises UnknownServiceError when asked for the 's3vectors' client.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.s3vectors.require_extra", lambda *a, **k: None)

    def _raise(service: str, **kwargs: Any) -> Any:
        raise RuntimeError("UnknownServiceError: Unknown service: 's3vectors'")

    module = types.ModuleType("boto3")
    module.client = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)

    with pytest.raises(StageError, match="boto3>=1.40"):
        S3VectorsStore(bucket="b", dim=2)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        S3VectorsStore(bucket="b", dim=2)
