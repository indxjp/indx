"""Unit tests for BigQueryStore — vendor SDK mocked, fully offline (coding-standards §11).

``google-cloud-bigquery`` is absent in this environment, so a fake ``google.cloud.bigquery``
module (with a tiny in-memory "table" and a ``Client`` that parses just the SQL the adapter
emits) is injected into ``sys.modules`` and the dependency gate is neutralized. Namespace
parents (``google``, ``google.cloud``) are registered too. The fake honors enough SQL to
exercise upsert (``MERGE``), cosine-ordered ``VECTOR_SEARCH``, and ``DELETE``.
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


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return 1.0 - dot / (na * nb)


class _ScalarQueryParameter:
    def __init__(self, name: str, type_: str, value: Any) -> None:
        self.name = name
        self.value = value


class _ArrayQueryParameter:
    def __init__(self, name: str, type_: str, values: Any) -> None:
        self.name = name
        self.value = list(values)


class _SchemaField:
    def __init__(self, name: str, field_type: str, mode: str = "NULLABLE") -> None:
        self.name = name
        self.field_type = field_type
        self.mode = mode


class _Table:
    def __init__(self, table_id: str, schema: list[_SchemaField]) -> None:
        self.table_id = table_id
        self.schema = schema


class _QueryJobConfig:
    def __init__(self, query_parameters: list[Any] | None = None) -> None:
        self.query_parameters = query_parameters or []


class _QueryJob:
    """A query job whose ``result()`` returns the canned rows for this SQL."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def result(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeClient:
    """In-memory BigQuery client keyed by id -> row dict. Parses only the SQL the adapter
    emits, by matching on stable substrings."""

    def __init__(self, project: str | None = None) -> None:
        self.project = project or "test-project"
        self.rows: dict[str, dict[str, Any]] = {}
        self.tables_created = 0
        self.fail_query = False

    def create_table(self, table: _Table, exists_ok: bool = False) -> _Table:
        self.tables_created += 1
        return table

    def query(self, sql: str, job_config: _QueryJobConfig | None = None) -> _QueryJob:
        if self.fail_query:
            raise RuntimeError("transient query failure")
        params = {p.name: p.value for p in (job_config.query_parameters if job_config else [])}
        if sql.startswith("MERGE"):
            self.rows[params["id"]] = {
                "id": params["id"],
                "content": params["content"],
                "doc_id": params["doc_id"],
                "position": params["position"],
                "prev_id": params["prev_id"],
                "next_id": params["next_id"],
                "embedding": params["embedding"],
            }
            return _QueryJob([])
        if "VECTOR_SEARCH" in sql:
            query, k = params["q"], params["k"]
            scored = sorted(
                self.rows.values(),
                key=lambda r: _cosine_distance(query, r["embedding"]),
            )
            out = [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "doc_id": r["doc_id"],
                    "position": r["position"],
                    "prev_id": r["prev_id"],
                    "next_id": r["next_id"],
                    "score": 1.0 - _cosine_distance(query, r["embedding"]),
                }
                for r in scored[:k]
            ]
            return _QueryJob(out)
        if sql.startswith("DELETE"):
            for cid in params["ids"]:
                self.rows.pop(cid, None)
            return _QueryJob([])
        raise AssertionError(f"unexpected SQL: {sql}")


class _FakeBigQuery(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("google.cloud.bigquery")
        self.last_client: _FakeClient | None = None
        self.Client = self._make_client
        self.QueryJobConfig = _QueryJobConfig
        self.ScalarQueryParameter = _ScalarQueryParameter
        self.ArrayQueryParameter = _ArrayQueryParameter
        self.SchemaField = _SchemaField
        self.Table = _Table

    def _make_client(self, project: str | None = None) -> _FakeClient:
        client = _FakeClient(project=project)
        self.last_client = client
        return client


@pytest.fixture
def fake_bigquery(monkeypatch: pytest.MonkeyPatch) -> _FakeBigQuery:
    """Inject a fake ``google.cloud.bigquery`` module and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.bigquery.require_extra", lambda *a, **k: None)

    bigquery = _FakeBigQuery()
    google_pkg = types.ModuleType("google")
    google_cloud_pkg = types.ModuleType("google.cloud")
    google_cloud_pkg.bigquery = bigquery  # type: ignore[attr-defined]

    # Namespace-package mocking: register every parent so ``from google.cloud import
    # bigquery`` resolves (coding-standards §11).
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bigquery)
    return bigquery


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)


def _store(fake_bigquery: _FakeBigQuery, **kwargs: Any) -> Any:
    from indx.store.bigquery import BigQueryStore

    return BigQueryStore(project="p", dataset="ds", table="tbl", **kwargs)


def test_satisfies_vector_store_protocol(fake_bigquery: _FakeBigQuery) -> None:
    from indx.store.base import VectorStore

    assert isinstance(_store(fake_bigquery, dim=2), VectorStore)


def test_init_creates_table_when_dim_given(fake_bigquery: _FakeBigQuery) -> None:
    _store(fake_bigquery, dim=2)
    assert fake_bigquery.last_client is not None
    assert fake_bigquery.last_client.tables_created == 1


def test_upsert_then_search_descending_order(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])
    hits = store.search([1.0, 0.0], k=2)

    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Round-trip: core types out, provenance preserved.
    assert hits[0].chunk.text == "a"
    assert hits[0].chunk.doc_id == "d"


def test_lazy_sizes_on_first_upsert(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery)  # no dim -> table creation deferred
    assert fake_bigquery.last_client is not None
    assert fake_bigquery.last_client.tables_created == 0
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    assert store.dim == 3
    assert fake_bigquery.last_client.tables_created == 1
    hits = store.search([1.0, 0.0, 0.0], k=1)
    assert [h.chunk.id for h in hits] == ["a"]


def test_search_before_sizing_raises(fake_bigquery: _FakeBigQuery) -> None:
    from indx.errors import StageError

    store = _store(fake_bigquery)  # no dim, nothing upserted
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=5)


def test_upsert_replaces_on_merge(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    replacement = Chunk(id="a", doc_id="d", position=1, text="updated", embedding=[0.0, 1.0])
    store.upsert([replacement])

    hits = store.search([0.0, 1.0], k=5)
    assert [h.chunk.id for h in hits] == ["a"]
    assert hits[0].chunk.text == "updated"
    assert hits[0].chunk.position == 1


def test_upsert_skips_unembedded_chunks(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([Chunk(id="a", doc_id="d", position=0, text="a", embedding=None)])
    assert store.search([1.0, 0.0], k=5) == []


def test_upsert_empty_is_noop(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([])
    assert store.search([1.0, 0.0], k=5) == []


def test_delete_removes_chunk(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["b"]


def test_delete_empty_is_noop(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery, dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    store.delete([])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["a"]


def test_upsert_wraps_failure_in_stage_error(fake_bigquery: _FakeBigQuery) -> None:
    from indx.errors import StageError

    store = _store(fake_bigquery, dim=2)
    assert fake_bigquery.last_client is not None
    fake_bigquery.last_client.fail_query = True
    with pytest.raises(StageError):
        store.upsert([_chunk("a", [1.0, 0.0])])


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise. Reload the
    # module first so the gate is the real require_extra (other tests monkeypatch the
    # module-level binding, and we must not depend on cross-test cleanup order).
    import importlib

    import indx.store.bigquery as bigquery_mod

    importlib.reload(bigquery_mod)
    with pytest.raises(MissingExtraError):
        bigquery_mod.BigQueryStore(project="p")
