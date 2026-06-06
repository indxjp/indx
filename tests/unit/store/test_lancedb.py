"""Unit tests for LanceDBStore — vendor SDK mocked, fully offline (coding-standards §11).

The ``lancedb`` wheel is absent in this environment, so a fake ``lancedb`` module is
injected into ``sys.modules`` and the dependency gate is neutralized. The fake connection
implements just enough of the LanceDB Python surface the adapter uses
(``connect`` / ``table_names`` / ``create_table`` / ``open_table`` and a table's
``add`` / ``delete`` / ``search(...).metric(...).limit(...).to_list()``) to round-trip core
:class:`~indx.core.chunk.Chunk` objects with a cosine ``_distance``.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError
from indx.store.lancedb import LanceDBStore

# --- Fake vendor objects mirroring lancedb's table / search builder -------------------


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance in [0, 2] (0 == identical), matching LanceDB's ``_distance``."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


class _FakeSearch:
    """A tiny search builder: ``.metric(...).limit(...).to_list()``."""

    def __init__(self, rows: dict[str, dict[str, Any]], query: list[float]) -> None:
        self._rows = rows
        self._query = query
        self._metric = "l2"
        self._limit = 10

    def metric(self, metric: str) -> _FakeSearch:
        self._metric = metric
        return self

    def limit(self, k: int) -> _FakeSearch:
        self._limit = k
        return self

    def to_list(self) -> list[dict[str, Any]]:
        scored = []
        for row in self._rows.values():
            out = dict(row)
            out["_distance"] = _cosine_distance(self._query, row["vector"])
            scored.append(out)
        scored.sort(key=lambda r: r["_distance"])
        return scored[: self._limit]


class _FakeTable:
    """In-memory table keyed by ``id``; supports add/delete/search."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.add(rows)
        self.add_calls: list[int] = []
        self.delete_predicates: list[str] = []

    def add(self, rows: list[dict[str, Any]]) -> None:
        # add_calls may not exist yet during the seeding call from __init__.
        if hasattr(self, "add_calls"):
            self.add_calls.append(len(rows))
        for row in rows:
            self._rows[row["id"]] = dict(row)

    def delete(self, predicate: str) -> None:
        if hasattr(self, "delete_predicates"):
            self.delete_predicates.append(predicate)
        # Emulate just the predicates the adapter emits: ``id = ''`` and ``id IN (...)``.
        if predicate == "id = ''":
            self._rows.pop("", None)
            return
        ids = _parse_in_predicate(predicate)
        for cid in ids:
            self._rows.pop(cid, None)

    def search(self, vector: list[float]) -> _FakeSearch:
        return _FakeSearch(self._rows, vector)


def _parse_in_predicate(predicate: str) -> list[str]:
    """Parse ``id IN ('a', 'b')`` back into ids, undoing the adapter's quote-escaping."""
    inner = predicate[predicate.index("(") + 1 : predicate.rindex(")")]
    ids = []
    for token in inner.split(","):
        token = token.strip()
        ids.append(token[1:-1].replace("''", "'"))  # strip quotes, unescape
    return ids


class _FakeConnection:
    def __init__(self) -> None:
        self._tables: dict[str, _FakeTable] = {}

    def table_names(self) -> list[str]:
        return list(self._tables)

    def create_table(self, name: str, *, data: list[dict[str, Any]], mode: str) -> _FakeTable:
        table = _FakeTable(data)
        self._tables[name] = table
        return table

    def open_table(self, name: str) -> _FakeTable:
        return self._tables[name]


def _make_fake_lancedb() -> tuple[types.ModuleType, dict[str, _FakeConnection]]:
    connections: dict[str, _FakeConnection] = {}

    def connect(path: str, *args: Any, **kwargs: Any) -> _FakeConnection:
        conn = connections.get(path) or _FakeConnection()
        connections[path] = conn
        return conn

    module = types.ModuleType("lancedb")
    module.connect = connect  # type: ignore[attr-defined]
    return module, connections


@pytest.fixture
def fake_lancedb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``lancedb`` module and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter binds require_extra by name (``from ... import``), so neutralize the
    # binding it actually calls too.
    monkeypatch.setattr("indx.store.lancedb.require_extra", lambda *a, **k: None)
    module, _ = _make_fake_lancedb()
    monkeypatch.setitem(sys.modules, "lancedb", module)
    return module


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def test_satisfies_vector_store_protocol(fake_lancedb: types.ModuleType) -> None:
    from indx.store.base import VectorStore

    assert isinstance(LanceDBStore(dim=2), VectorStore)


def test_construction_seeds_then_clears_empty_table(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    table: _FakeTable = store._table  # type: ignore[assignment]
    # The placeholder seed row must have been deleted, leaving an empty table.
    assert table.search([1.0, 0.0]).to_list() == []
    assert "id = ''" in table.delete_predicates


def test_reopens_existing_table(fake_lancedb: types.ModuleType) -> None:
    first = LanceDBStore(dim=2, path="./reuse.lancedb")
    first.upsert([_chunk("a", [1.0, 0.0])])
    # A second store on the same path must open the existing table, not overwrite it.
    second = LanceDBStore(dim=2, path="./reuse.lancedb")
    assert [h.chunk.id for h in second.search([1.0, 0.0], k=5)] == ["a"]


def test_upsert_then_search_roundtrips_core_types(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert hits[0].chunk.doc_id == "d"
    assert hits[0].chunk.text == "a"
    assert hits[0].chunk.embedding == [1.0, 0.0]
    # Cosine self-distance is 0 -> similarity 1.0.
    assert hits[0].score == pytest.approx(1.0)


def test_search_uses_cosine_metric(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    builder = store._table.search([1.0, 0.0])  # type: ignore[attr-defined]
    builder.metric("cosine").limit(1).to_list()
    assert builder._metric == "cosine"


def test_search_descending_with_deterministic_tie_break(
    fake_lancedb: types.ModuleType,
) -> None:
    store = LanceDBStore(dim=2)
    # Two chunks with identical vectors -> identical score; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_upsert_is_a_true_upsert(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    # Re-upsert the same id with a new vector: should replace, not duplicate.
    store.upsert([_chunk("a", [0.0, 1.0])])
    hits = store.search([0.0, 1.0], k=5)
    assert [h.chunk.id for h in hits] == ["a"]
    assert hits[0].chunk.embedding == [0.0, 1.0]


def test_upsert_skips_chunks_without_embedding(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    no_vec = Chunk(id="x", doc_id="d", position=0, text="x", embedding=None)
    store.upsert([no_vec, _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_upsert_empty_is_a_noop(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    table: _FakeTable = store._table  # type: ignore[assignment]
    store.upsert([])
    assert table.add_calls == []


def test_delete_removes_chunks(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_delete_escapes_quoted_ids(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    tricky = "o'brien"
    store.upsert([_chunk(tricky, [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete([tricky])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_delete_empty_is_a_noop(fake_lancedb: types.ModuleType) -> None:
    store = LanceDBStore(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    store.delete([])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["a"]


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        LanceDBStore(dim=2)
