"""Unit tests for PgVectorStore — vendor SDK mocked, fully offline (coding-standards §11).

``psycopg`` and ``pgvector`` are absent in this environment, so a fake ``psycopg`` module
(with a tiny in-memory "database") and a fake ``pgvector.psycopg.register_vector`` are
injected into ``sys.modules`` and the dependency gate is neutralized. The fake honors just
enough SQL to exercise upsert (``ON CONFLICT``), cosine-ordered search, and delete.
"""

from __future__ import annotations

import math
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError

if TYPE_CHECKING:
    from collections.abc import Sequence


class _Jsonb:
    """Stand-in for ``psycopg.types.json.Jsonb``: just carries the wrapped dict."""

    def __init__(self, obj: dict[str, Any]) -> None:
        self.obj = obj


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConnection:
    """In-memory store keyed by id -> (embedding, payload dict). Parses only the SQL the
    adapter emits, by matching on stable substrings."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self.commits = 0
        self.rollbacks = 0
        self.registered = False
        self.fail_executemany = False

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _FakeCursor:
        params = params or ()
        if "CREATE EXTENSION" in sql or "CREATE TABLE" in sql:
            return _FakeCursor([])
        if sql.startswith("SELECT"):
            query, _, k = params
            scored = sorted(
                (
                    (cid, payload, 1.0 - _cosine_distance(query, emb))
                    for cid, (emb, payload) in self.rows.items()
                ),
                key=lambda r: _cosine_distance(query, self.rows[r[0]][0]),
            )
            return _FakeCursor([(cid, payload, score) for cid, payload, score in scored[:k]])
        if sql.startswith("DELETE"):
            (ids,) = params
            for cid in ids:
                self.rows.pop(cid, None)
            return _FakeCursor([])
        raise AssertionError(f"unexpected SQL: {sql}")

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        assert "INSERT" in sql and "ON CONFLICT" in sql
        if self.fail_executemany:
            raise RuntimeError("transient write failure")
        for cid, embedding, jsonb in rows:
            # ON CONFLICT DO UPDATE: last write wins, mirroring upsert semantics.
            self.rows[cid] = (list(embedding), dict(jsonb.obj))

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakePsycopg(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("psycopg")
        self.last_conn: _FakeConnection | None = None

    def connect(self, dsn: str) -> _FakeConnection:
        conn = _FakeConnection()
        self.last_conn = conn
        return conn


@pytest.fixture
def fake_pgvector(monkeypatch: pytest.MonkeyPatch) -> _FakePsycopg:
    """Inject fake ``psycopg`` / ``pgvector`` modules and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.pgvector.require_extra", lambda *a, **k: None)

    psycopg = _FakePsycopg()
    json_mod = types.ModuleType("psycopg.types.json")
    json_mod.Jsonb = _Jsonb  # type: ignore[attr-defined]
    types_mod = types.ModuleType("psycopg.types")
    types_mod.json = json_mod  # type: ignore[attr-defined]

    def _register_vector(conn: _FakeConnection) -> None:
        conn.registered = True

    pgvector_pkg = types.ModuleType("pgvector")
    pgvector_psycopg = types.ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = _register_vector  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", types_mod)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", json_mod)
    monkeypatch.setitem(sys.modules, "pgvector", pgvector_pkg)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg)
    return psycopg


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)


def test_satisfies_vector_store_protocol(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.base import VectorStore
    from indx.store.pgvector import PgVectorStore

    assert isinstance(PgVectorStore("postgresql://x", dim=2), VectorStore)


def test_init_registers_vector_and_creates_table(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    PgVectorStore("postgresql://x", dim=2)
    assert fake_pgvector.last_conn is not None
    assert fake_pgvector.last_conn.registered is True


def test_upsert_then_search_descending_order(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])
    hits = store.search([1.0, 0.0], k=2)

    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Round-trip: core types out, provenance preserved.
    assert hits[0].chunk.text == "a"
    assert hits[0].chunk.doc_id == "d"


def test_upsert_replaces_on_conflict(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    # Re-upsert same id with a new vector/text — ON CONFLICT should replace, not duplicate.
    replacement = Chunk(id="a", doc_id="d", position=1, text="updated", embedding=[0.0, 1.0])
    store.upsert([replacement])

    hits = store.search([0.0, 1.0], k=5)
    assert [h.chunk.id for h in hits] == ["a"]
    assert hits[0].chunk.text == "updated"
    assert hits[0].chunk.position == 1


def test_upsert_skips_unembedded_chunks(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([Chunk(id="a", doc_id="d", position=0, text="a", embedding=None)])
    assert store.search([1.0, 0.0], k=5) == []


def test_upsert_empty_is_noop(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([])
    assert store.search([1.0, 0.0], k=5) == []


def test_delete_removes_chunk(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["b"]


def test_delete_empty_is_noop(fake_pgvector: _FakePsycopg) -> None:
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    store.delete([])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["a"]


def test_upsert_rolls_back_on_write_failure(fake_pgvector: _FakePsycopg) -> None:
    from indx.errors import StageError
    from indx.store.pgvector import PgVectorStore

    store = PgVectorStore("postgresql://x", dim=2)
    conn = fake_pgvector.last_conn
    assert conn is not None
    # A transient write error must roll back the aborted psycopg3 tx, otherwise every
    # later statement on this connection fails with "current transaction is aborted".
    conn.fail_executemany = True
    with pytest.raises(StageError):
        store.upsert([_chunk("a", [1.0, 0.0])])
    assert conn.rollbacks == 1


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise. Reload the
    # module first so the gate is the real require_extra (other tests monkeypatch the
    # module-level binding, and we must not depend on cross-test cleanup order).
    import importlib

    import indx.store.pgvector as pgvector_mod

    importlib.reload(pgvector_mod)
    with pytest.raises(MissingExtraError):
        pgvector_mod.PgVectorStore("postgresql://x")
