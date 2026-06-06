"""Unit tests for :class:`indx.embed.e5.E5Embedder` with the vendor SDK mocked.

The ``e5`` extra (sentence-transformers + torch) is absent in the offline suite, so we inject a
fake ``sentence_transformers`` module into ``sys.modules`` and neutralize the dependency gate
(coding-standards §11). Tests are deterministic and offline.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.base import Embedder
from indx.embed.e5 import E5Embedder
from indx.errors import MissingExtraError

if TYPE_CHECKING:
    from collections.abc import Sequence

_DIM = 768


class _FakeSentenceTransformer:
    """Stand-in for ``sentence_transformers.SentenceTransformer``.

    Records constructor args and returns deterministic per-text vectors so tests can assert shape
    and ordering without any real model or network.
    """

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device

    def get_embedding_dimension(self) -> int:
        return _DIM

    def get_sentence_embedding_dimension(self) -> int:  # legacy name kept for fallback coverage
        return _DIM

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 12,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        # Deterministic: vector i is filled with float(i) so order is verifiable downstream.
        return [[float(i)] * _DIM for i, _ in enumerate(texts)]


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the extra gate and inject a fake ``sentence_transformers`` module."""
    # e5.py imported require_extra by name, so patch the reference it actually holds.
    monkeypatch.setattr("indx.embed.e5.require_extra", lambda *a, **k: None)
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_satisfies_embedder_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    embedder = E5Embedder()
    assert isinstance(embedder, Embedder)


def test_records_model_id_as_name_and_resolves_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    embedder = E5Embedder("intfloat/e5-base-v2")
    assert embedder.name == "intfloat/e5-base-v2"
    assert embedder.dim == _DIM


def test_embed_round_trips_one_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    embedder = E5Embedder()
    texts = ["alpha", "beta", "gamma"]
    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == embedder.dim
        assert all(isinstance(v, float) for v in vec)
    # Input order is preserved (fake encodes text i as all-float(i)).
    assert vectors[0][0] == 0.0
    assert vectors[1][0] == 1.0
    assert vectors[2][0] == 2.0


def test_embed_empty_returns_empty_without_calling_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake(monkeypatch)
    embedder = E5Embedder()

    def _boom(*_a: Any, **_k: Any) -> list[list[float]]:
        raise AssertionError("encode must not be called on an empty batch")

    monkeypatch.setattr(embedder._model, "encode", _boom)
    assert embedder.embed([]) == []


def test_missing_extra_raises_on_construction() -> None:
    # Do NOT neutralize require_extra: in the dep-absent environment, constructing must fail loud.
    with pytest.raises(MissingExtraError) as excinfo:
        E5Embedder()
    assert excinfo.value.extra == "e5"
    assert "pip install indx[e5]" in str(excinfo.value)
