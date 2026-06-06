"""The pipeline forwards the embedder's real ``dim`` to pre-sizing stores (T10 P1 bug).

A vector store that pre-sizes its collection/column at construction (qdrant / pgvector /
lancedb) must be told the *actual* embedding width up front. The earlier code hard-coded
``dim=768`` in the store and never plumbed the embedder's dimensionality through, so an
openai (1536-dim) or bge (1024-dim) vector failed on upsert into the default-768 collection.

These tests are fully offline and mock-based: a spy ``get_store`` records the ``dim`` kwarg
it is constructed with, and a tiny fake embedder advertises a known ``dim``. No heavy SDK is
imported — the registry's ``get_store`` is monkeypatched at the pipeline's binding so no real
qdrant/pgvector wheel is required. Mirrors the SDK-surface style in
``tests/unit/test_pipeline_sdk_api.py`` and the spy-store style in ``tests/unit/store/``.
"""

from __future__ import annotations

from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.embed.hash_embedder import HashEmbedder
from indx.pipeline import pipeline as pipeline_mod
from indx.pipeline.pipeline import DirectoryPipeline, _build_store
from indx.store.base import SearchHit


class _FakeEmbedder:
    """Embedder protocol stand-in with a fixed, declared dimensionality."""

    def __init__(self, *, name: str = "fake", dim: int = 1536) -> None:
        self.name = name
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class _SpyStore:
    """Records the constructor kwargs (notably ``dim``) the pipeline built it with."""

    name = "qdrant"
    last_dim: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.dim = kwargs.get("dim")
        _SpyStore.last_dim = self.dim

    def upsert(self, chunks: list[Chunk]) -> None:  # pragma: no cover - not exercised
        return None

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:  # pragma: no cover
        return []

    def delete(self, chunk_ids: list[str]) -> None:  # pragma: no cover
        return None


def _spy_get_store(
    monkeypatch: pytest.MonkeyPatch,
    backend: str = "qdrant",
) -> dict[str, Any]:
    """Patch ``get_store`` in the pipeline module with a spy; return the captured kwargs."""
    captured: dict[str, Any] = {}

    def fake_get_store(name: str, **kwargs: Any) -> _SpyStore:
        captured["name"] = name
        captured["kwargs"] = kwargs
        store = _SpyStore(**kwargs)
        store.name = backend
        return store

    monkeypatch.setattr(pipeline_mod, "get_store", fake_get_store)
    return captured


# ---------------------------------------------------------------- _build_store unit-level


def test_build_store_injects_embedder_dim_for_pre_sizing_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    _build_store("qdrant", embedder=_FakeEmbedder(dim=1536))
    assert captured["kwargs"].get("dim") == 1536


def test_build_store_uses_hash_256_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    _build_store("qdrant", embedder=HashEmbedder())  # default dim is 256
    assert captured["kwargs"].get("dim") == 256


def test_build_store_skips_dim_for_lazy_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # jsonl/chroma size lazily from the first vector and reject a ``dim`` kwarg.
    captured = _spy_get_store(monkeypatch, backend="jsonl")
    _build_store("jsonl", embedder=_FakeEmbedder(dim=1536))
    assert "dim" not in captured["kwargs"]


def test_build_store_no_embedder_means_no_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    # Embed disabled: nothing is upserted, so the store keeps its own default dim.
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    _build_store("qdrant", embedder=None)
    assert "dim" not in captured["kwargs"]


def test_explicit_config_dim_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit [store.qdrant] dim must NOT be overwritten by the embedder's dim.
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    _build_store("qdrant", embedder=_FakeEmbedder(dim=1536), options={"dim": 64})
    assert captured["kwargs"].get("dim") == 64


# ---------------------------------------------------------------- full pipeline __init__


def test_pipeline_passes_embedder_dim_to_store(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder=_FakeEmbedder(dim=1536),
        store="qdrant",
        output="jsonl",
    )
    assert captured["kwargs"].get("dim") == 1536


def test_pipeline_hash_dim_flows_to_store(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
    )
    assert captured["kwargs"].get("dim") == 256


def test_pipeline_embed_disabled_passes_no_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
        embed=False,
    )
    assert "dim" not in captured["kwargs"]


def test_use_rebinds_store_with_new_embedder_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _spy_get_store(monkeypatch, backend="qdrant")
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",  # 256
        store="qdrant",
        output="jsonl",
    )
    assert captured["kwargs"].get("dim") == 256
    # Rebinding embedder + store in one call must size the new store to the new embedder.
    pipe.use(embedder=_FakeEmbedder(dim=1024), store="qdrant")
    assert captured["kwargs"].get("dim") == 1024
