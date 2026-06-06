"""Unit tests for :class:`indx.store.chroma.ChromaStore`.

The ``chromadb`` SDK is intentionally absent in this environment, so it is mocked:
a fake ``chromadb`` module is injected into ``sys.modules`` and the dependency gate
(:func:`indx.utils.lazy.require_extra`) is neutralized so construction proceeds.
The fake collection implements a tiny brute-force cosine index so ``upsert`` /
``search`` / ``delete`` round-trip core types exactly like the real backend.
One test deliberately leaves the gate intact to assert the friendly
:class:`~indx.errors.MissingExtraError` in the real (dep-absent) environment.

All tests are deterministic and offline.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.errors import MissingExtraError, StageError
from indx.store.base import SearchHit, VectorStore
from indx.store.chroma import ChromaStore


class _FakeCollection:
    """In-memory stand-in for a Chroma collection (parallel-list API)."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for cid, emb, doc, md in zip(ids, embeddings, documents, metadatas, strict=True):
            self.records[cid] = {"embedding": emb, "document": doc, "metadata": md}

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        vector = query_embeddings[0]
        scored = [
            (cid, rec, _cosine(vector, rec["embedding"])) for cid, rec in self.records.items()
        ]
        # Chroma returns DISTANCE (lower is better); emulate by distance = 1 - cosine.
        scored.sort(key=lambda t: (-t[2], t[0]))
        top = scored[:n_results]
        return {
            "ids": [[cid for cid, _, _ in top]],
            "documents": [[rec["document"] for _, rec, _ in top]],
            "metadatas": [[rec["metadata"] for _, rec, _ in top]],
            "distances": [[1.0 - sim for _, _, sim in top]],
        }

    def delete(self, *, ids: list[str]) -> None:
        for cid in ids:
            self.records.pop(cid, None)


class _FakePersistentClient:
    """Stand-in for ``chromadb.PersistentClient`` that hands out a fake collection."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        *,
        embedding_function: Any = None,
        configuration: dict[str, Any] | None = None,
    ) -> _FakeCollection:
        self.last_configuration = configuration
        return self._collections.setdefault(name, _FakeCollection())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _install_fake_chromadb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Neutralize the extra gate and inject a fake ``chromadb`` module.

    Returns:
        The fake ``chromadb`` module so tests can introspect it if needed.
    """
    monkeypatch.setattr("indx.store.chroma.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("chromadb")
    fake_module.PersistentClient = _FakePersistentClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", fake_module)
    return fake_module


def _chunk(cid: str, vec: list[float], *, position: int = 0) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=position, text=cid, embedding=vec)


def test_satisfies_vector_store_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter is recognized as a :class:`VectorStore` at runtime."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    assert isinstance(store, VectorStore)
    assert store.name == "chroma"


def test_collection_created_with_cosine_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collection is created with cosine space, not Chroma's default L2."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    client: _FakePersistentClient = store._client  # type: ignore[assignment]
    assert client.last_configuration == {"hnsw": {"space": "cosine"}}


def test_upsert_search_round_trips_in_descending_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """``search`` returns SearchHits wrapping core Chunks, highest score first."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)

    assert all(isinstance(h, SearchHit) for h in hits)
    assert all(isinstance(h.chunk, Chunk) for h in hits)
    assert [h.chunk.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score


def test_search_converts_metadata_back_to_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty-string neighbor sentinels round-trip back to None on read."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([Chunk(id="x", doc_id="doc1", position=3, text="body", embedding=[1.0, 0.0])])

    hit = store.search([1.0, 0.0], k=1)[0]

    assert hit.chunk.id == "x"
    assert hit.chunk.doc_id == "doc1"
    assert hit.chunk.position == 3
    assert hit.chunk.text == "body"
    assert hit.chunk.prev_id is None
    assert hit.chunk.next_id is None


def test_neighbor_ids_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-empty prev/next ids survive the metadata round-trip."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert(
        [
            Chunk(
                id="m",
                doc_id="d",
                position=1,
                text="mid",
                prev_id="l",
                next_id="r",
                embedding=[1.0, 0.0],
            )
        ]
    )

    hit = store.search([1.0, 0.0], k=1)[0]
    assert hit.chunk.prev_id == "l"
    assert hit.chunk.next_id == "r"


def test_upsert_skips_unembedded_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk with no embedding is never pushed to the backend."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([_chunk("a", [1.0, 0.0]), Chunk(id="b", doc_id="d", position=1, text="b")])

    collection: _FakeCollection = store._col  # type: ignore[assignment]
    assert set(collection.records) == {"a"}


def test_delete_removes_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleted ids no longer appear in search results."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([_chunk("a", [1.0, 0.0])])
    store.delete(["a"])
    assert store.search([1.0, 0.0], k=5) == []


def test_empty_upsert_and_delete_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty batches do not touch the backend."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([])
    store.delete([])
    collection: _FakeCollection = store._col  # type: ignore[assignment]
    assert collection.records == {}


def test_query_failure_wrapped_in_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vendor exception during query is normalized to a typed StageError."""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()

    def _boom(**_: Any) -> Any:
        raise RuntimeError("backend down")

    store._col.query = _boom  # type: ignore[method-assign, union-attr]
    with pytest.raises(StageError):
        store.search([1.0, 0.0], k=1)


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction raises MissingExtraError.

    The dependency gate is NOT neutralized here, and ``chromadb`` is genuinely not
    installed, so ``require_extra`` raises before any vendor import.
    """
    assert "chromadb" not in sys.modules or sys.modules.get("chromadb") is None
    with pytest.raises(MissingExtraError):
        ChromaStore()
