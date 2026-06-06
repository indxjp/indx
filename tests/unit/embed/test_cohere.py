"""Unit tests for :class:`indx.embed.cohere.CohereEmbedder`.

The ``cohere`` SDK is intentionally absent in this environment, so it is mocked:
a fake ``cohere`` module is injected into ``sys.modules`` and the dependency gate
(:func:`indx.utils.lazy.require_extra`) is neutralized so construction proceeds.
One test deliberately leaves the gate intact to assert the friendly
:class:`~indx.errors.MissingExtraError` in the real (dep-absent) environment.

All tests are deterministic and offline.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.base import Embedder
from indx.embed.cohere import CohereEmbedder
from indx.errors import MissingExtraError, StageError

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeEmbedResponse:
    """Mimics the object returned by ``cohere.Client.embed`` (has ``.embeddings``)."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeClient:
    """Stand-in for ``cohere.Client`` that records calls and returns fixed vectors."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []

    def embed(
        self,
        *,
        texts: Sequence[str],
        model: str,
        input_type: str,
    ) -> _FakeEmbedResponse:
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        # Deterministic: one 1024-dim vector per text, encoding its index.
        vectors = [[float(i)] * 1024 for i, _ in enumerate(texts)]
        return _FakeEmbedResponse(vectors)


def _install_fake_cohere(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Neutralize the extra gate and inject a fake ``cohere`` module.

    Returns:
        The fake ``Client`` class so tests can assert on recorded calls.
    """
    monkeypatch.setattr("indx.embed.cohere.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("cohere")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    monkeypatch.setenv("COHERE_API_KEY", "test-key-not-a-real-secret")
    return _FakeClient


def test_satisfies_embedder_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter is recognized as an :class:`Embedder` at runtime."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder()
    assert isinstance(embedder, Embedder)


def test_defaults_model_and_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default model is embed-english-v3.0 with the matching 1024 dimension."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder()
    assert embedder.name == "embed-english-v3.0"
    assert embedder.dim == 1024


def test_light_model_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A light v3 model resolves to its 384-dim output width."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder(model="embed-english-light-v3.0")
    assert embedder.name == "embed-english-light-v3.0"
    assert embedder.dim == 384


def test_embed_round_trips_one_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed`` returns one core vector of length ``dim`` per input, in order."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder()
    texts = ["alpha", "bravo", "charlie"]

    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    for vector in vectors:
        assert len(vector) == embedder.dim
        assert all(isinstance(v, float) for v in vector)
    # Order preserved: fake encodes the input index into every component.
    assert vectors[0][0] == 0.0
    assert vectors[1][0] == 1.0
    assert vectors[2][0] == 2.0


def test_embed_passes_model_and_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured model and input_type are forwarded to the vendor client."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder(input_type="search_query")
    embedder.embed(["one", "two"])

    client: _FakeClient = embedder._client  # type: ignore[assignment]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "embed-english-v3.0"
    assert call["input_type"] == "search_query"
    assert call["texts"] == ["one", "two"]


def test_embed_empty_input_skips_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch returns ``[]`` without calling the backend."""
    _install_fake_cohere(monkeypatch)
    embedder = CohereEmbedder()
    assert embedder.embed([]) == []
    client: _FakeClient = embedder._client  # type: ignore[assignment]
    assert client.calls == []


def test_explicit_api_key_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit api_key argument is passed to the client, no env needed."""
    monkeypatch.setattr("indx.embed.cohere.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("cohere")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    embedder = CohereEmbedder(api_key="explicit-key")
    client: _FakeClient = embedder._client  # type: ignore[assignment]
    assert client.api_key == "explicit-key"


def test_missing_api_key_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key (arg or env) raises a typed StageError."""
    monkeypatch.setattr("indx.embed.cohere.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("cohere")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(StageError):
        CohereEmbedder()


def test_backend_failure_wrapped_in_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vendor exception is normalized to a typed StageError."""
    monkeypatch.setattr("indx.embed.cohere.require_extra", lambda *a, **k: None)

    class _BoomClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def embed(self, **_: Any) -> Any:
            raise RuntimeError("upstream 503")

    fake_module = types.ModuleType("cohere")
    fake_module.Client = _BoomClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)
    monkeypatch.setenv("COHERE_API_KEY", "k")

    embedder = CohereEmbedder()
    with pytest.raises(StageError):
        embedder.embed(["x"])


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction raises MissingExtraError.

    The dependency gate is NOT neutralized here, and ``cohere`` is genuinely not
    installed, so ``require_extra`` raises before any vendor import.
    """
    assert "cohere" not in sys.modules or sys.modules.get("cohere") is None
    with pytest.raises(MissingExtraError):
        CohereEmbedder()
