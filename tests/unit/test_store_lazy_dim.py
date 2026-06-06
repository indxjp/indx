"""F3 — pre-sizing stores must tolerate an unknown ``dim`` and size LAZILY (T10).

Wave-2 plumbed ``embedder.dim`` through the pipeline, but a store an SDK user constructs
*directly* without a dim (e.g. ``QdrantStore()``) used to default to a hard-coded width and
create the collection at ``__init__`` — mis-sizing it. The fix: when ``dim`` is supplied the
collection/table is created up-front (unchanged pipeline path); when ``dim`` is ``None`` the
backend defers creation until the first :meth:`upsert` and sizes it to ``len(first_vector)``.
A :meth:`search` before any upsert with no dim raises a clean typed :class:`IndxError`.

All three pre-sizing stores (qdrant / pgvector / lancedb) are exercised with the vendor SDK
mocked into ``sys.modules`` (the wheels are absent, env is offline) and ``require_extra``
neutralized, mirroring the per-store unit tests. The fakes RECORD the width each backend
sizes its collection/table to, so the assertions pin sizing to the vector — not a guess.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import IndxError


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


# --- Qdrant ---------------------------------------------------------------------------


class _QPointStruct:
    def __init__(self, *, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _QVectorParams:
    def __init__(self, *, size: int, distance: Any) -> None:
        self.size = size
        self.distance = distance


class _QDistance:
    COSINE = "Cosine"


class _QPointIdsList:
    def __init__(self, *, points: list[str]) -> None:
        self.points = points


class _QScoredPoint:
    def __init__(
        self, *, id: str, score: float, payload: dict[str, Any], vector: list[float]
    ) -> None:
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector


class _QQueryResponse:
    def __init__(self, points: list[_QScoredPoint]) -> None:
        self.points = points


class _QFakeModels:
    PointStruct = _QPointStruct
    VectorParams = _QVectorParams
    Distance = _QDistance
    PointIdsList = _QPointIdsList


class _FakeQdrantClient:
    """In-memory stand-in that RECORDS every ``create_collection`` size it is asked for."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self._points: dict[str, _QPointStruct] = {}
        self.collections: set[str] = set()
        self.create_sizes: list[int] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, *, vectors_config: Any) -> None:
        self.collections.add(name)
        self.create_sizes.append(vectors_config.size)

    def upsert(self, collection_name: str, *, points: list[_QPointStruct], wait: bool) -> None:
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
    ) -> _QQueryResponse:
        pts = [
            _QScoredPoint(id=p.id, score=1.0, payload=dict(p.payload), vector=list(p.vector))
            for p in self._points.values()
        ]
        return _QQueryResponse(pts[:limit])

    def delete(self, collection_name: str, *, points_selector: _QPointIdsList) -> None:
        for pid in points_selector.points:
            self._points.pop(pid, None)


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.qdrant.require_extra", lambda *a, **k: None)
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    module.models = _QFakeModels  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)


def test_qdrant_with_dim_creates_sized_collection_at_construction(fake_qdrant: None) -> None:
    # (a) Existing behavior: an explicit dim sizes the collection up-front (pipeline path).
    from indx.store.qdrant import QdrantStore

    store = QdrantStore(dim=1536)
    client: _FakeQdrantClient = store._client
    assert client.create_sizes == [1536]
    assert "indx" in client.collections


def test_qdrant_no_dim_defers_then_sizes_to_first_vector(fake_qdrant: None) -> None:
    # (b) No dim: construction must NOT create the collection; the first upsert sizes it to
    # the real vector width — here 1024, not any hard-coded default.
    from indx.store.qdrant import QdrantStore

    store = QdrantStore()
    client: _FakeQdrantClient = store._client
    assert client.create_sizes == []  # deferred — nothing sized yet
    assert "indx" not in client.collections

    store.upsert([_chunk("a", [0.0] * 1024)])
    # (c) Byte/vector width comes from the upserted vector, not a 768 (or any) default.
    assert client.create_sizes == [1024]
    assert "indx" in client.collections


def test_qdrant_search_before_sizing_raises_typed_error(fake_qdrant: None) -> None:
    from indx.store.qdrant import QdrantStore

    store = QdrantStore()  # no dim, nothing upserted
    with pytest.raises(IndxError):
        store.search([0.0, 1.0])


def test_qdrant_lazy_then_roundtrips(fake_qdrant: None) -> None:
    from indx.store.qdrant import QdrantStore

    store = QdrantStore()
    store.upsert([_chunk("a", [1.0, 0.0, 0.0]), _chunk("b", [0.0, 1.0, 0.0])])
    hits = store.search([1.0, 0.0, 0.0], k=2)
    assert {h.chunk.id for h in hits} == {"a", "b"}


# --- pgvector -------------------------------------------------------------------------


class _PgJsonb:
    def __init__(self, obj: dict[str, Any]) -> None:
        self.obj = obj


class _PgCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakePgConnection:
    """Records the ``vector(<dim>)`` width every ``CREATE TABLE`` asks for."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self.create_dims: list[int] = []

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _PgCursor:
        params = params or ()
        if "CREATE EXTENSION" in sql:
            return _PgCursor([])
        if "CREATE TABLE" in sql:
            # Extract the integer inside ``vector(<n>)``.
            inside = sql.split("vector(", 1)[1].split(")", 1)[0]
            self.create_dims.append(int(inside))
            return _PgCursor([])
        if sql.startswith("SELECT"):
            return _PgCursor([(cid, payload, 1.0) for cid, (_, payload) in self.rows.items()])
        if sql.startswith("DELETE"):
            (ids,) = params
            for cid in ids:
                self.rows.pop(cid, None)
            return _PgCursor([])
        raise AssertionError(f"unexpected SQL: {sql}")

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        for cid, embedding, jsonb in rows:
            self.rows[cid] = (list(embedding), dict(jsonb.obj))

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@pytest.fixture
def fake_pgvector(monkeypatch: pytest.MonkeyPatch) -> _FakePgConnection:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.pgvector.require_extra", lambda *a, **k: None)

    conn = _FakePgConnection()
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda *a, **k: conn  # type: ignore[attr-defined]
    json_mod = types.ModuleType("psycopg.types.json")
    json_mod.Jsonb = _PgJsonb  # type: ignore[attr-defined]
    types_mod = types.ModuleType("psycopg.types")
    types_mod.json = json_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", types_mod)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", json_mod)

    pgvector_pkg = types.ModuleType("pgvector")
    pgvector_psycopg = types.ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = lambda *a, **k: None  # type: ignore[attr-defined]
    pgvector_pkg.psycopg = pgvector_psycopg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pgvector", pgvector_pkg)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg)
    return conn


def test_pgvector_with_dim_sizes_table_at_construction(fake_pgvector: _FakePgConnection) -> None:
    from indx.store.pgvector import PgVectorStore

    PgVectorStore("dsn://x", dim=1536)
    assert fake_pgvector.create_dims == [1536]


def test_pgvector_no_dim_defers_then_sizes_to_first_vector(
    fake_pgvector: _FakePgConnection,
) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("dsn://x")
    assert fake_pgvector.create_dims == []  # deferred
    store.upsert([_chunk("a", [0.0] * 1024)])
    assert fake_pgvector.create_dims == [1024]  # width from the vector, not a default


def test_pgvector_search_before_sizing_raises_typed_error(
    fake_pgvector: _FakePgConnection,
) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("dsn://x")
    with pytest.raises(IndxError):
        store.search([0.0, 1.0])


# --- lancedb --------------------------------------------------------------------------


class _FakeLanceTable:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        # The seed row fixes the vector width — record it like a real schema would.
        self.seed_dim = len(data[0]["vector"]) if data else 0
        self.rows: dict[str, dict[str, Any]] = {}

    def delete(self, predicate: str) -> None:
        if predicate == "id = ''":
            return
        # Best-effort: clear any ids named in the IN(...) predicate.
        for cid in list(self.rows):
            if f"'{cid}'" in predicate:
                self.rows.pop(cid, None)

    def add(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            self.rows[r["id"]] = r

    def search(self, vector: list[float]) -> _FakeLanceSearch:
        return _FakeLanceSearch(list(self.rows.values()))


class _FakeLanceSearch:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def metric(self, _m: str) -> _FakeLanceSearch:
        return self

    def limit(self, k: int) -> _FakeLanceSearch:
        self._rows = self._rows[:k]
        return self

    def to_list(self) -> list[dict[str, Any]]:
        return [{**r, "_distance": 0.0} for r in self._rows]


class _FakeLanceDB:
    def __init__(self) -> None:
        self.tables: dict[str, _FakeLanceTable] = {}
        self.create_dims: list[int] = []

    def table_names(self) -> list[str]:
        return list(self.tables)

    def open_table(self, name: str) -> _FakeLanceTable:
        return self.tables[name]

    def create_table(self, name: str, *, data: list[dict[str, Any]], mode: str) -> _FakeLanceTable:
        table = _FakeLanceTable(data)
        self.create_dims.append(table.seed_dim)
        self.tables[name] = table
        return table


@pytest.fixture
def fake_lancedb(monkeypatch: pytest.MonkeyPatch) -> _FakeLanceDB:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.lancedb.require_extra", lambda *a, **k: None)
    db = _FakeLanceDB()
    module = types.ModuleType("lancedb")
    module.connect = lambda *a, **k: db  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lancedb", module)
    return db


def test_lancedb_with_dim_seeds_table_at_construction(fake_lancedb: _FakeLanceDB) -> None:
    from indx.store.lancedb import LanceDBStore

    LanceDBStore(dim=1536)
    assert fake_lancedb.create_dims == [1536]


def test_lancedb_no_dim_defers_then_sizes_to_first_vector(fake_lancedb: _FakeLanceDB) -> None:
    from indx.store.lancedb import LanceDBStore

    store = LanceDBStore()
    assert fake_lancedb.create_dims == []  # deferred — no table seeded yet
    store.upsert([_chunk("a", [0.0] * 1024)])
    assert fake_lancedb.create_dims == [1024]  # width from the vector, not a default


def test_lancedb_search_before_sizing_raises_typed_error(fake_lancedb: _FakeLanceDB) -> None:
    from indx.store.lancedb import LanceDBStore

    store = LanceDBStore()
    with pytest.raises(IndxError):
        store.search([0.0, 1.0])
