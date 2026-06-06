"""Unit tests for :mod:`indx.embed.litellm` with the vendor ``litellm`` SDK mocked.

Injects a fake ``litellm`` exposing ``embedding`` and neutralizes ``require_extra`` so the
adapter constructs offline. Asserts the ``Embedder`` protocol, input-order batching, dimension
discovery from the first response, and the dep-absent ``MissingExtraError`` contract.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.embed.litellm  # noqa: F401  -- force-load before any require_extra patch
from indx.embed.base import Embedder
from indx.embed.litellm import LiteLLMEmbedder
from indx.errors import MissingExtraError, StageError


class _Response:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.embed.litellm.require_extra", lambda *a, **k: None)
    module = types.ModuleType("litellm")
    module.last_request = {}  # type: ignore[attr-defined]

    def embedding(**kwargs: Any) -> _Response:
        module.last_request = kwargs  # type: ignore[attr-defined]
        # Return a 4-dim vector per input, dict-shaped like LiteLLM.
        return _Response(
            [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": i} for i in range(len(kwargs["input"]))]
        )

    module.embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def test_satisfies_embedder_protocol(fake_litellm: types.ModuleType) -> None:
    assert isinstance(LiteLLMEmbedder(), Embedder)


def test_embeds_one_vector_per_input(fake_litellm: types.ModuleType) -> None:
    vectors = LiteLLMEmbedder(model="bedrock/amazon.titan-embed-text-v2:0").embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)
    assert fake_litellm.last_request["model"] == "bedrock/amazon.titan-embed-text-v2:0"


def test_dimension_discovered_from_first_response(fake_litellm: types.ModuleType) -> None:
    emb = LiteLLMEmbedder()
    assert emb.dim == 1536  # fallback before any call
    emb.embed(["hello"])
    assert emb.dim == 4  # corrected to the real width


def test_pinned_dim_is_not_overwritten(fake_litellm: types.ModuleType) -> None:
    emb = LiteLLMEmbedder(dim=4096)
    emb.embed(["hello"])
    assert emb.dim == 4096


def test_name_records_actual_model(fake_litellm: types.ModuleType) -> None:
    assert LiteLLMEmbedder(model="azure/my-deployment").name == "azure/my-deployment"


def test_empty_input_returns_empty(fake_litellm: types.ModuleType) -> None:
    assert LiteLLMEmbedder().embed([]) == []


def test_backend_error_becomes_stage_error(
    fake_litellm: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs: Any) -> None:
        raise RuntimeError("provider down")

    monkeypatch.setattr(fake_litellm, "embedding", boom)
    with pytest.raises(StageError):
        LiteLLMEmbedder().embed(["x"])


def test_construction_without_extra_raises_missing_extra() -> None:
    with pytest.raises(MissingExtraError):
        LiteLLMEmbedder()
