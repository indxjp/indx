"""Regression: resource-owning stores release their vendor client on close().

Confirms the resource-leak fix for the remote vector stores that retain (or, in
Azure's case, deliberately do not retain) a vendor client:

* :class:`~indx.store.opensearch.OpenSearchStore` holds an ``opensearchpy.OpenSearch``
  client whose ``RequestsHttpConnection`` pool must be released; ``close()`` calls the
  vendor ``close()`` and drops the reference, is idempotent, and is wired as a context
  manager.
* :class:`~indx.store.azure_search.AzureAISearchStore` keeps no persistent client on
  ``self`` (a fresh client is built per call), so its ``close()`` is an explicit,
  idempotent no-op that still completes the context-manager contract.

Fully offline: the absent ``opensearchpy`` / ``boto3`` / ``azure.*`` wheels are faked
into ``sys.modules`` and the dependency gate is neutralized (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.store.azure_search import AzureAISearchStore
from indx.store.opensearch import OpenSearchStore

# --- OpenSearch fakes -----------------------------------------------------------------


class _RequestsHttpConnection:  # marker class; only identity is used by the adapter
    pass


class _AWSV4SignerAuth:
    def __init__(self, credentials: Any, region: Any, service: str) -> None:
        self.service = service


class _FakeIndices:
    def __init__(self) -> None:
        self.names: set[str] = set()

    def exists(self, *, index: str) -> bool:
        return index in self.names

    def create(self, *, index: str, body: Any) -> None:
        self.names.add(index)


class _FakeOpenSearch:
    """In-memory stand-in that records whether the vendor ``close()`` was called."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.indices = _FakeIndices()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeSession:
    def get_credentials(self) -> str:
        return "fake-credentials"


@pytest.fixture
def fake_opensearch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.opensearch.require_extra", lambda *a, **k: None)

    ospy = types.ModuleType("opensearchpy")
    ospy.OpenSearch = _FakeOpenSearch  # type: ignore[attr-defined]
    ospy.AWSV4SignerAuth = _AWSV4SignerAuth  # type: ignore[attr-defined]
    ospy.RequestsHttpConnection = _RequestsHttpConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opensearchpy", ospy)

    boto3 = types.ModuleType("boto3")
    boto3.Session = _FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)


def _opensearch_store() -> OpenSearchStore:
    return OpenSearchStore(host="abc.us-east-1.aoss.amazonaws.com", region="us-east-1", dim=2)


def test_opensearch_close_calls_vendor_close_and_drops_reference(
    fake_opensearch: None,
) -> None:
    store = _opensearch_store()
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    store.close()
    # Vendor pool released exactly once and the held reference is dropped.
    assert client.close_calls == 1
    assert store._client is None


def test_opensearch_close_is_idempotent(fake_opensearch: None) -> None:
    store = _opensearch_store()
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    store.close()
    store.close()  # second call must not raise nor re-close the (dropped) client
    assert client.close_calls == 1
    assert store._client is None


def test_opensearch_context_manager_closes_on_exit(fake_opensearch: None) -> None:
    with _opensearch_store() as store:
        client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    assert client.close_calls == 1
    assert store._client is None


# --- Azure fakes (store retains no client; close() is a documented no-op) --------------


class _AzureKeyCredential:
    def __init__(self, key: str) -> None:
        self.key = key


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.store.azure_search.require_extra", lambda *a, **k: None)

    modules = {
        "azure": _make_module("azure"),
        "azure.core": _make_module("azure.core"),
        "azure.core.credentials": _make_module(
            "azure.core.credentials", AzureKeyCredential=_AzureKeyCredential
        ),
        "azure.search": _make_module("azure.search"),
        "azure.search.documents": _make_module("azure.search.documents", SearchClient=object),
        "azure.search.documents.indexes": _make_module(
            "azure.search.documents.indexes", SearchIndexClient=object
        ),
        "azure.search.documents.indexes.models": _make_module(
            "azure.search.documents.indexes.models"
        ),
        "azure.search.documents.models": _make_module("azure.search.documents.models"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _azure_store() -> AzureAISearchStore:
    return AzureAISearchStore(
        endpoint="https://svc.search.windows.net", key="secret", index_name="indx", dim=2
    )


def test_azure_close_is_idempotent_noop(fake_azure: None) -> None:
    store = _azure_store()
    # No persistent client is retained; close() must succeed and be repeatable.
    store.close()
    store.close()
    assert store.close() is None


def test_azure_context_manager_yields_store(fake_azure: None) -> None:
    with _azure_store() as store:
        assert isinstance(store, AzureAISearchStore)
