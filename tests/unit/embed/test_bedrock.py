"""Unit tests for :class:`indx.embed.bedrock.BedrockEmbedder`.

The ``boto3`` SDK is intentionally absent in this environment, so it is mocked: a fake
``boto3`` module is injected into ``sys.modules`` and the dependency gate
(:func:`indx.utils.lazy.require_extra`) is neutralized so construction proceeds. One test
deliberately leaves the gate intact to assert the friendly
:class:`~indx.errors.MissingExtraError` in the real (dep-absent) environment.

All tests are deterministic and offline.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from indx.embed.base import Embedder
from indx.embed.bedrock import BedrockEmbedder
from indx.errors import MissingExtraError, StageError


class _FakeBody:
    """Mimics the streaming ``["body"]`` object: ``.read()`` yields JSON bytes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeClient:
    """Stand-in for ``boto3.client('bedrock-runtime')`` recording invoke_model calls."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:  # noqa: N803
        request = json.loads(body)
        self.calls.append({"modelId": modelId, "body": request})
        if modelId.startswith("cohere."):
            n = len(request["texts"])
            return {"body": _FakeBody({"embeddings": {"float": [[1.0] * self.dim] * n}})}
        # Titan: one inputText per call; honor the requested dimensions.
        width = request["dimensions"]
        return {"body": _FakeBody({"embedding": [1.0] * width})}


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, *, dim: int = 1024) -> _FakeClient:
    """Neutralize the extra gate and inject a fake ``boto3`` module.

    Returns:
        The single fake client instance so tests can assert on recorded calls.
    """
    monkeypatch.setattr("indx.embed.bedrock.require_extra", lambda *a, **k: None)
    client = _FakeClient(dim=dim)
    fake_module = types.ModuleType("boto3")
    fake_module.client = lambda *a, **k: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)
    return client


def test_satisfies_embedder_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter is recognized as an :class:`Embedder` at runtime."""
    _install_fake_boto3(monkeypatch)
    embedder = BedrockEmbedder()
    assert isinstance(embedder, Embedder)


def test_defaults_model_and_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default model is Titan v2 with the matching 1024 dimension; name == model."""
    _install_fake_boto3(monkeypatch)
    embedder = BedrockEmbedder()
    assert embedder.name == "amazon.titan-embed-text-v2:0"
    assert embedder.model == "amazon.titan-embed-text-v2:0"
    assert embedder.dim == 1024


def test_titan_dimensions_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Titan v2 ``dimensions`` knob (512/256) is reflected in ``dim``."""
    _install_fake_boto3(monkeypatch, dim=512)
    embedder = BedrockEmbedder(dimensions=512)
    assert embedder.dim == 512


def test_invalid_dimensions_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dimensions value Titan v2 does not accept raises a typed StageError."""
    _install_fake_boto3(monkeypatch)
    with pytest.raises(StageError):
        BedrockEmbedder(dimensions=999)


def test_embed_round_trips_one_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed`` returns one core vector of length ``dim`` per input, in order."""
    client = _install_fake_boto3(monkeypatch)
    embedder = BedrockEmbedder()
    texts = ["alpha", "bravo", "charlie"]

    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    for vector in vectors:
        assert len(vector) == embedder.dim
        assert all(isinstance(v, float) for v in vector)
    # Titan embeds one inputText per call: one invoke_model per text.
    assert len(client.calls) == len(texts)
    assert client.calls[0]["body"]["inputText"] == "alpha"
    assert client.calls[0]["body"]["dimensions"] == 1024
    assert client.calls[0]["body"]["normalize"] is True


def test_embed_parses_embedding_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter extracts ``out['embedding']`` from the JSON body and floats it."""
    _install_fake_boto3(monkeypatch, dim=256)
    embedder = BedrockEmbedder(dimensions=256)

    vectors = embedder.embed(["one"])

    assert len(vectors) == 1
    assert vectors[0] == [1.0] * 256


def test_cohere_branch_batches_and_parses_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cohere-on-Bedrock sends the whole batch and reads ``embeddings['float']``."""
    client = _install_fake_boto3(monkeypatch)
    embedder = BedrockEmbedder(model="cohere.embed-english-v3", input_type="search_query")

    vectors = embedder.embed(["one", "two"])

    assert len(vectors) == 2
    assert all(len(v) == 1024 for v in vectors)
    # One batched call carrying texts + input_type.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == "cohere.embed-english-v3"
    assert call["body"]["texts"] == ["one", "two"]
    assert call["body"]["input_type"] == "search_query"


def test_embed_empty_input_skips_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch returns ``[]`` without calling the backend."""
    client = _install_fake_boto3(monkeypatch)
    embedder = BedrockEmbedder()
    assert embedder.embed([]) == []
    assert client.calls == []


def test_backend_failure_wrapped_in_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vendor exception is normalized to a typed StageError."""
    monkeypatch.setattr("indx.embed.bedrock.require_extra", lambda *a, **k: None)

    class _BoomClient:
        def invoke_model(self, **_: Any) -> Any:
            raise RuntimeError("upstream 503")

    fake_module = types.ModuleType("boto3")
    fake_module.client = lambda *a, **k: _BoomClient()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)

    embedder = BedrockEmbedder()
    with pytest.raises(StageError):
        embedder.embed(["x"])


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction raises MissingExtraError.

    The dependency gate is NOT neutralized here, and ``boto3`` is genuinely not
    installed, so ``require_extra`` raises before any vendor import.
    """
    assert "boto3" not in sys.modules or sys.modules.get("boto3") is None
    with pytest.raises(MissingExtraError):
        BedrockEmbedder()
