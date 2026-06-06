"""Mocked unit tests for :class:`indx.embed.azure.AzureOpenAIEmbedder`.

The ``openai`` SDK is not installed in this environment, so it is faked: a stand-in ``openai``
module is injected into ``sys.modules`` and the dependency gate is neutralized. Tests are
deterministic and offline (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.azure import AzureOpenAIEmbedder
from indx.errors import MissingExtraError

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeEmbedding:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data: list[_FakeEmbedding]) -> None:
        self.data = data


class _FakeEmbeddings:
    """Returns a deterministic vector per input; encodes ``len(text)`` to verify order."""

    def __init__(self, client: _FakeAzureOpenAI) -> None:
        self._client = client

    def create(self, *, model: str, input: Sequence[str], **kwargs: Any) -> _FakeResponse:
        self._client.calls.append({"model": model, "input": list(input), **kwargs})
        width = kwargs.get("dimensions", AzureOpenAIEmbedder.dim)
        data = [_FakeEmbedding([float(len(text))] * width) for text in input]
        return _FakeResponse(data)


class _FakeAzureOpenAI:
    """Minimal stand-in for ``openai.AzureOpenAI`` that records call kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.embeddings = _FakeEmbeddings(self)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAzureOpenAI]:
    """Neutralize the extra gate and inject a fake ``openai`` module."""
    # azure.py binds require_extra at import time (``from ... import require_extra``), so patch
    # the name in both the source module and the consuming module.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.embed.azure.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("openai")
    fake_module.AzureOpenAI = _FakeAzureOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return _FakeAzureOpenAI


def _make_embedder(**overrides: Any) -> AzureOpenAIEmbedder:
    kwargs: dict[str, Any] = {
        "model": "text-embedding-3-small",
        "api_key": "secret-key",
        "azure_endpoint": "https://example.openai.azure.com",
    }
    kwargs.update(overrides)
    return AzureOpenAIEmbedder(**kwargs)


def test_satisfies_embedder_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    from indx.embed.base import Embedder

    _install_fake_openai(monkeypatch)
    embedder = _make_embedder()
    assert isinstance(embedder, Embedder)


def test_embed_returns_one_vector_per_text_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    embedder = _make_embedder()
    texts = ["a", "bb", "ccc"]

    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == embedder.dim for v in vectors)
    # Fake encodes len(text) into each component, so input order is verifiable.
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]
    # The request used the deployment as the model and forwarded the texts unchanged.
    client = embedder._ensure_client()  # cached fake
    assert isinstance(client, _FakeAzureOpenAI)
    (call,) = client.calls
    assert call["model"] == "text-embedding-3-small"
    assert call["input"] == texts
    assert "dimensions" not in call  # no override → SDK uses the deployment default


def test_client_constructed_with_azure_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    embedder = _make_embedder(api_version="2024-10-21")
    client = embedder._ensure_client()
    assert isinstance(client, _FakeAzureOpenAI)
    assert client.init_kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert client.init_kwargs["api_key"] == "secret-key"
    assert client.init_kwargs["api_version"] == "2024-10-21"


def test_embed_empty_returns_empty_without_calling_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    embedder = _make_embedder()
    assert embedder.embed([]) == []
    # No client was even built (and so no API call was made).
    assert embedder._client is None


def test_name_records_actual_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    assert _make_embedder(model="my-embed-deployment").name == "my-embed-deployment"


def test_dim_inferred_from_deployment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    # 3-large is 3072-dim; 3-small/ada are 1536-dim; unknown deployments default to 1536.
    assert _make_embedder(model="text-embedding-3-large").dim == 3072
    assert _make_embedder(model="text-embedding-3-small").dim == 1536
    assert _make_embedder(model="my-ada-002").dim == 1536
    assert _make_embedder(model="custom-deployment").dim == 1536


def test_dimensions_override_forwarded_and_sizes_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    embedder = _make_embedder(model="text-embedding-3-large", dimensions=512)
    assert embedder.dim == 512  # override wins over inference

    vectors = embedder.embed(["a"])

    assert len(vectors[0]) == 512
    (call,) = embedder._ensure_client().calls  # type: ignore[attr-defined]
    assert call["dimensions"] == 512


def test_embed_wraps_backend_failure_in_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from indx.errors import StageError

    _install_fake_openai(monkeypatch)
    embedder = _make_embedder()

    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("azure exploded")

    monkeypatch.setattr(embedder._ensure_client().embeddings, "create", _boom)
    with pytest.raises(StageError) as excinfo:
        embedder.embed(["x"])
    assert excinfo.value.stage == "embed"


def test_missing_deployment_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from indx.errors import StageError

    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_EMBED_DEPLOYMENT", raising=False)
    embedder = AzureOpenAIEmbedder(api_key="k", azure_endpoint="https://e.openai.azure.com")
    with pytest.raises(StageError) as excinfo:
        embedder.embed(["x"])
    assert excinfo.value.stage == "embed"


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        AzureOpenAIEmbedder(model="dep")
