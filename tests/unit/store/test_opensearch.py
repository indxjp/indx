"""Unit tests for OpenSearchStore — vendor SDK mocked, fully offline (coding-standards §11).

The ``opensearchpy`` and ``boto3`` wheels are absent in this environment, so fake modules
are injected into ``sys.modules`` and the dependency gate is neutralized. The fake client
implements just enough of the opensearch-py surface the adapter uses
(``indices.exists`` / ``indices.create`` / ``index`` / ``search`` / ``delete_by_query``)
to round-trip core :class:`~indx.core.chunk.Chunk` objects.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError
from indx.store.opensearch import OpenSearchStore

# --- Fake vendor objects mirroring opensearchpy / boto3 -------------------------------


class _RequestsHttpConnection:  # marker class, only identity is used by the adapter
    pass


class _AWSV4SignerAuth:
    def __init__(self, credentials: Any, region: Any, service: str) -> None:
        self.credentials = credentials
        self.region = region
        self.service = service


class _FakeIndices:
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.bodies: dict[str, Any] = {}

    def exists(self, *, index: str) -> bool:
        return index in self.names

    def create(self, *, index: str, body: Any) -> None:
        self.names.add(index)
        self.bodies[index] = body


class _FakeOpenSearch:
    """In-memory stand-in: stores docs and answers a deterministic cosine search."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.indices = _FakeIndices()
        self._docs: list[dict[str, Any]] = []

    def index(self, *, index: str, body: dict[str, Any]) -> None:
        self._docs.append(dict(body))

    def bulk(self, body: Any, *args: Any, **kwargs: Any) -> Any:
        # opensearchpy.helpers.bulk calls client.bulk(...) with a serialized payload; the
        # fake instead receives the already-expanded actions from the helper stub below.
        raise NotImplementedError  # pragma: no cover

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        query = body["query"]["knn"]["embedding"]["vector"]
        k = body["query"]["knn"]["embedding"]["k"]
        scored = [
            {"_score": _cosine(query, d["embedding"]), "_source": dict(d)} for d in self._docs
        ]
        scored.sort(key=lambda h: -h["_score"])
        return {"hits": {"hits": scored[:k]}}

    def delete_by_query(self, *, index: str, body: dict[str, Any], refresh: bool = False) -> None:
        self.last_delete_refresh = refresh
        ids = set(body["query"]["terms"]["chunk_id"])
        self._docs = [d for d in self._docs if d["chunk_id"] not in ids]


def _make_bulk(counter: dict[str, int]):
    """Build a fake ``opensearchpy.helpers.bulk`` that records its invocation count.

    The real helper sends one ``_bulk`` HTTP request per call; here we just ingest each
    action's ``_source`` into the fake client's docs and bump a counter so a test can
    assert one bulk call per batch rather than one request per document.
    """

    def bulk(client: Any, actions: Any, *args: Any, **kwargs: Any) -> tuple[int, list[Any]]:
        counter["calls"] += 1
        acts = list(actions)
        for action in acts:
            client._docs.append(dict(action["_source"]))
        return len(acts), []

    return bulk


class _FakeSession:
    def get_credentials(self) -> str:
        return "fake-credentials"


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@pytest.fixture
def fake_opensearch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject fake ``opensearchpy`` / ``boto3`` modules and neutralize the gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra`` at import time, binding
    # its own module-level name; patch that binding too so neutralization is effective
    # regardless of import order.
    monkeypatch.setattr("indx.store.opensearch.require_extra", lambda *a, **k: None)

    ospy = types.ModuleType("opensearchpy")
    ospy.OpenSearch = _FakeOpenSearch  # type: ignore[attr-defined]
    ospy.AWSV4SignerAuth = _AWSV4SignerAuth  # type: ignore[attr-defined]
    ospy.RequestsHttpConnection = _RequestsHttpConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opensearchpy", ospy)

    # opensearchpy.helpers.bulk is imported lazily inside upsert(); provide a fake that
    # records how many times it is called so a test can assert true per-batch batching.
    helpers = types.ModuleType("opensearchpy.helpers")
    bulk_calls: dict[str, int] = {"calls": 0}
    helpers.bulk = _make_bulk(bulk_calls)  # type: ignore[attr-defined]
    ospy.helpers = helpers  # type: ignore[attr-defined]
    ospy.bulk_calls = bulk_calls  # type: ignore[attr-defined]  # surfaced for assertions
    monkeypatch.setitem(sys.modules, "opensearchpy.helpers", helpers)

    boto3 = types.ModuleType("boto3")
    boto3.Session = _FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    return ospy


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(
        id=cid, doc_id="d", position=0, text=cid, prev_id=None, next_id=None, embedding=vec
    )


def _store(**kwargs: Any) -> OpenSearchStore:
    kwargs.setdefault("host", "abc.us-east-1.aoss.amazonaws.com")
    kwargs.setdefault("region", "us-east-1")
    return OpenSearchStore(**kwargs)


def test_satisfies_vector_store_protocol(fake_opensearch: types.ModuleType) -> None:
    from indx.store.base import VectorStore

    assert isinstance(_store(dim=2), VectorStore)


def test_construction_creates_sized_index(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    assert "indx" in client.indices.names
    mapping = client.indices.bodies["indx"]["mappings"]["properties"]["embedding"]
    assert mapping["type"] == "knn_vector"
    assert mapping["dimension"] == 2
    assert mapping["space_type"] == "cosinesimil"


def test_sigv4_service_name_is_aoss_not_es(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    auth = client.init_kwargs["http_auth"]
    assert auth.service == "aoss"
    assert auth.region == "us-east-1"
    # SSL and the SigV4 connection class are wired as the design requires.
    assert client.init_kwargs["use_ssl"] is True
    assert client.init_kwargs["hosts"] == [
        {"host": "abc.us-east-1.aoss.amazonaws.com", "port": 443}
    ]


def test_upsert_then_search_roundtrips_core_types(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score
    # Core types in, core types out — no vendor object leaks.
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert hits[0].chunk.doc_id == "d"
    assert hits[0].chunk.text == "a"
    assert hits[0].chunk.embedding == [1.0, 0.0]


def test_lazy_sizing_from_first_vector(fake_opensearch: types.ModuleType) -> None:
    # No dim at construction: the index is created on the first upsert, sized to the
    # first real vector's width (T10), never a guess.
    store = _store()
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    assert "indx" not in client.indices.names
    store.upsert([_chunk("a", [1.0, 0.0, 0.0])])
    mapping = client.indices.bodies["indx"]["mappings"]["properties"]["embedding"]
    assert mapping["dimension"] == 3


def test_search_descending_with_deterministic_tie_break(
    fake_opensearch: types.ModuleType,
) -> None:
    store = _store(dim=2)
    # Two chunks with identical vectors -> identical score; id breaks the tie.
    store.upsert([_chunk("z", [1.0, 0.0]), _chunk("a", [1.0, 0.0])])
    hits = store.search([1.0, 0.0], k=5)
    assert [h.chunk.id for h in hits] == ["a", "z"]


def test_upsert_skips_chunks_without_embedding(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    no_vec = Chunk(id="x", doc_id="d", position=0, text="x", embedding=None)
    store.upsert([no_vec, _chunk("y", [1.0, 0.0])])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["y"]


def test_upsert_is_idempotent_by_chunk_id(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    store.upsert([_chunk("a", [0.0, 1.0])])  # re-upsert same id with a new vector
    hits = store.search([0.0, 1.0], k=5)
    assert [h.chunk.id for h in hits] == ["a"]
    assert hits[0].chunk.embedding == [0.0, 1.0]


def test_delete_removes_documents(fake_opensearch: types.ModuleType) -> None:
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    store.delete(["a"])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["b"]


def test_upsert_uses_one_bulk_request_per_batch(fake_opensearch: types.ModuleType) -> None:
    # True batching: opensearchpy.helpers.bulk must be invoked once per _BATCH slice, not
    # once per document. With 600 chunks and _BATCH=256 that is ceil(600/256) == 3 calls.
    from indx.store.opensearch import _BATCH

    store = _store(dim=2)
    n = 2 * _BATCH + 50
    store.upsert([_chunk(f"c{i}", [1.0, 0.0]) for i in range(n)])
    expected = -(-n // _BATCH)  # ceil division
    assert fake_opensearch.bulk_calls["calls"] == expected  # type: ignore[attr-defined]
    assert expected < n  # sanity: fewer requests than documents


def test_upsert_refreshes_delete_before_reindex(fake_opensearch: types.ModuleType) -> None:
    # The delete-then-index path must close the AOSS eventual-consistency window by asking
    # delete_by_query to refresh, otherwise a re-upsert can leave stale duplicates.
    store = _store(dim=2)
    store.upsert([_chunk("a", [1.0, 0.0])])
    client: _FakeOpenSearch = store._client  # type: ignore[assignment]
    assert client.last_delete_refresh is True


def test_search_before_sizing_raises(fake_opensearch: types.ModuleType) -> None:
    from indx.errors import StageError

    store = _store()  # no dim, nothing upserted yet
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=5)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        _store(dim=2)
