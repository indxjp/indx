"""Unit tests for AzureAISearchStore — vendor SDK mocked, fully offline (coding-standards §11).

The ``azure-search-documents`` wheel is absent in this environment, so fake ``azure.*``
namespace modules are injected into ``sys.modules`` and the dependency gate is neutralized.
The fakes implement just enough of the Azure AI Search surface the adapter uses
(``SearchIndexClient.create_or_update_index`` / ``SearchClient.merge_or_upload_documents`` /
``search`` / ``delete_documents``) to round-trip core :class:`~indx.core.chunk.Chunk` objects.
Every parent of the ``azure.*`` namespace package is registered in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError, StageError
from indx.store.azure_search import AzureAISearchStore

# --- Fake vendor objects mirroring the azure-search-documents surface ------------------

# A single shared in-memory index so SearchIndexClient and SearchClient (constructed
# separately, per-call, by the adapter) see the same documents.
_STATE: dict[str, Any] = {"documents": {}, "created_index": None, "init_kwargs": []}


class _AzureKeyCredential:
    def __init__(self, key: str) -> None:
        self.key = key


class _SearchFieldDataType:
    String = "Edm.String"
    Single = "Edm.Single"
    Int32 = "Edm.Int32"

    @staticmethod
    def Collection(t: str) -> str:  # noqa: N802 - mirrors the vendor classmethod name
        return f"Collection({t})"


class _SearchField:
    def __init__(self, *, name: str, type: str, **kwargs: Any) -> None:
        self.name = name
        self.type = type
        self.kwargs = kwargs


class _VectorSearchProfile:
    def __init__(self, *, name: str, algorithm_configuration_name: str) -> None:
        self.name = name
        self.algorithm_configuration_name = algorithm_configuration_name


class _HnswAlgorithmConfiguration:
    def __init__(self, *, name: str) -> None:
        self.name = name


class _VectorSearch:
    def __init__(self, *, profiles: list[Any], algorithms: list[Any]) -> None:
        self.profiles = profiles
        self.algorithms = algorithms


class _SearchIndex:
    def __init__(self, *, name: str, fields: list[Any], vector_search: Any) -> None:
        self.name = name
        self.fields = fields
        self.vector_search = vector_search


class _VectorizedQuery:
    def __init__(self, *, vector: list[float], k_nearest_neighbors: int, fields: str) -> None:
        self.vector = vector
        self.k_nearest_neighbors = k_nearest_neighbors
        self.fields = fields


class _SearchIndexClient:
    def __init__(self, *, endpoint: str, credential: Any) -> None:
        _STATE["init_kwargs"].append({"endpoint": endpoint, "credential": credential})

    def create_or_update_index(self, index: _SearchIndex) -> None:
        _STATE["created_index"] = index


class _SearchClient:
    def __init__(self, *, endpoint: str, index_name: str, credential: Any) -> None:
        self.index_name = index_name

    def merge_or_upload_documents(self, documents: list[dict[str, Any]]) -> None:
        for d in documents:
            _STATE["documents"][d["id"]] = dict(d)

    def search(
        self, *, search_text: Any, vector_queries: list[_VectorizedQuery], select: list[str]
    ) -> list[dict[str, Any]]:
        q = vector_queries[0]
        scored = []
        for d in _STATE["documents"].values():
            row = {key: d.get(key) for key in select}
            row["@search.score"] = _cosine(q.vector, d["contentVector"])
            scored.append(row)
        scored.sort(key=lambda r: -r["@search.score"])
        return scored[: q.k_nearest_neighbors]

    def delete_documents(self, documents: list[dict[str, Any]]) -> None:
        for d in documents:
            _STATE["documents"].pop(d["id"], None)


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def fake_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``azure.*`` namespace modules and neutralize the dependency gate."""
    _STATE["documents"] = {}
    _STATE["created_index"] = None
    _STATE["init_kwargs"] = []

    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra`` at import time, binding
    # its own module-level name; patch that binding too so neutralization is effective
    # regardless of import order.
    monkeypatch.setattr("indx.store.azure_search.require_extra", lambda *a, **k: None)

    # Namespace packages: register every parent in sys.modules.
    modules = {
        "azure": _make_module("azure"),
        "azure.core": _make_module("azure.core"),
        "azure.core.credentials": _make_module(
            "azure.core.credentials", AzureKeyCredential=_AzureKeyCredential
        ),
        "azure.search": _make_module("azure.search"),
        "azure.search.documents": _make_module(
            "azure.search.documents", SearchClient=_SearchClient
        ),
        "azure.search.documents.indexes": _make_module(
            "azure.search.documents.indexes", SearchIndexClient=_SearchIndexClient
        ),
        "azure.search.documents.indexes.models": _make_module(
            "azure.search.documents.indexes.models",
            SearchIndex=_SearchIndex,
            SearchField=_SearchField,
            SearchFieldDataType=_SearchFieldDataType,
            VectorSearch=_VectorSearch,
            VectorSearchProfile=_VectorSearchProfile,
            HnswAlgorithmConfiguration=_HnswAlgorithmConfiguration,
        ),
        "azure.search.documents.models": _make_module(
            "azure.search.documents.models", VectorizedQuery=_VectorizedQuery
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


_ENV = {
    "endpoint": "https://svc.search.windows.net",
    "key": "secret",
    "index_name": "indx",
}


def _store(dim: int | None = None) -> AzureAISearchStore:
    return AzureAISearchStore(dim=dim, **_ENV)


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def test_satisfies_vector_store_protocol(fake_azure: None) -> None:
    from indx.store.base import VectorStore

    assert isinstance(_store(dim=2), VectorStore)


def test_construction_creates_index_lazily_not_on_init(fake_azure: None) -> None:
    _store(dim=2)
    # The index is built on first upsert, not at construction.
    assert _STATE["created_index"] is None


def test_first_upsert_sizes_index_to_given_dim(fake_azure: None) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    index = _STATE["created_index"]
    assert index is not None
    vec_field = next(f for f in index.fields if f.name == "contentVector")
    assert vec_field.kwargs["vector_search_dimensions"] == 2
    assert vec_field.type == "Collection(Edm.Single)"


def test_lazy_dim_sizes_to_first_vector_width(fake_azure: None) -> None:
    store = _store(dim=None)  # dim unknown at construction
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    vec_field = next(f for f in _STATE["created_index"].fields if f.name == "contentVector")
    assert vec_field.kwargs["vector_search_dimensions"] == 3


def test_upsert_then_search_roundtrips_core_types(fake_azure: None) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert hits[0].chunk.doc_id == "d"
    assert hits[0].chunk.text == "a"


def test_search_descending_with_deterministic_tie_break(fake_azure: None) -> None:
    store = _store(dim=2)
    # Two chunks with identical vectors -> identical score; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_upsert_skips_chunks_without_embedding(fake_azure: None) -> None:
    store = _store(dim=2)
    no_vec = Chunk(id="x", doc_id="d", position=0, text="x", embedding=None)
    store.upsert([no_vec, _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_delete_removes_documents(fake_azure: None) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_search_before_sizing_raises_stage_error(fake_azure: None) -> None:
    store = _store(dim=None)  # never sized, nothing upserted
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=5)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        AzureAISearchStore(dim=2, **_ENV)
