"""Regression: BigQueryStore releases its vendor client on close().

:class:`~indx.store.bigquery.BigQueryStore` holds a ``google.cloud.bigquery.Client`` that
owns a pooled HTTP session (``requests``/``google-auth`` transport). A store that is never
closed leaks those connections until GC, and (because :class:`~indx.store.base.VectorStore`
is a ``@runtime_checkable`` ``Protocol`` that now declares ``close()``) a store missing
``close()`` no longer satisfies the contract the pipeline relies on
(``pipeline.close()`` calls ``store.close()``).

This proves ``close()`` calls the vendor ``close()``, drops the held reference, is
idempotent, and that the store stays a structural :class:`VectorStore`. Fully offline: the
absent ``google-cloud-bigquery`` wheel is faked into ``sys.modules`` and the dependency
gate is neutralized (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _FakeClient:
    """Minimal stand-in recording whether the vendor ``close()`` was called."""

    def __init__(self, project: str | None = None) -> None:
        self.project = project or "test-project"
        self.close_calls = 0

    def create_table(self, table: Any, exists_ok: bool = False) -> Any:
        return table

    def close(self) -> None:
        self.close_calls += 1


class _Table:
    def __init__(self, table_id: str, schema: list[Any]) -> None:
        self.table_id = table_id
        self.schema = schema


class _SchemaField:
    def __init__(self, name: str, field_type: str, mode: str = "NULLABLE") -> None:
        self.name = name
        self.field_type = field_type
        self.mode = mode


class _FakeBigQuery(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("google.cloud.bigquery")
        self.last_client: _FakeClient | None = None
        self.Client = self._make_client
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

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bigquery)
    return bigquery


def _store(fake_bigquery: _FakeBigQuery) -> Any:
    from indx.store.bigquery import BigQueryStore

    return BigQueryStore(project="p", dataset="ds", table="tbl", dim=2)


def test_close_calls_vendor_close_and_drops_reference(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery)
    client = store._client
    assert isinstance(client, _FakeClient)
    store.close()
    # Vendor session released exactly once and the held reference is dropped.
    assert client.close_calls == 1
    assert store._client is None


def test_close_is_idempotent(fake_bigquery: _FakeBigQuery) -> None:
    store = _store(fake_bigquery)
    client = store._client
    store.close()
    store.close()  # second call must not raise nor re-close the (dropped) client
    assert isinstance(client, _FakeClient)
    assert client.close_calls == 1
    assert store._client is None


def test_store_satisfies_vector_store_protocol(fake_bigquery: _FakeBigQuery) -> None:
    from indx.store.base import VectorStore

    # The runtime-checkable protocol now requires close(); the store must conform.
    assert isinstance(_store(fake_bigquery), VectorStore)
