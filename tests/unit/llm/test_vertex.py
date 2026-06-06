"""Mocked unit tests for :class:`indx.llm.vertex.VertexLLM`.

The ``google-genai`` SDK is not installed in this environment, so it is faked: stand-in
``google``, ``google.genai`` and ``google.genai.types`` modules are injected into
``sys.modules`` and the dependency gate is neutralized. Tests are deterministic and offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.llm.base import LLM
from indx.llm.vertex import VertexLLM


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self._client.calls.append(kwargs)
        return _FakeResponse(self._client.reply)


class _FakeClient:
    """Minimal stand-in for ``google.genai.Client`` that records call kwargs."""

    reply: str | None = "fake-vertex-response"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.models = _FakeModels(self)


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Neutralize the extra gate and inject a fake ``google.genai`` namespace package."""
    # vertex.py binds require_extra at import time (``from ... import require_extra``),
    # so patch the name in both the source module and the consuming module.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.vertex.require_extra", lambda *a, **k: None)

    # Namespace package: every parent must live in sys.modules.
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = _FakeClient  # type: ignore[attr-defined]
    types_module = types.ModuleType("google.genai.types")
    types_module.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]
    google_module.genai = genai_module  # type: ignore[attr-defined]
    genai_module.types = types_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    return _FakeClient


def test_satisfies_llm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    adapter = VertexLLM(project="my-project", location="us-central1")
    assert adapter.name == "vertex"
    assert isinstance(adapter, LLM)


def test_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    adapter = VertexLLM(project="p", location="l")
    assert adapter.model == "gemini-2.5-flash"


def test_complete_round_trips_through_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    adapter = VertexLLM(model="gemini-custom", project="my-project", location="us-central1")

    out = adapter.complete("hello", system="be brief", max_tokens=64, temperature=0.0)

    assert out == "fake-vertex-response"
    client = adapter._ensure_client()  # cached fake
    assert isinstance(client, _FakeClient)
    # Client constructed in Vertex AI mode with project/location.
    assert client.init_kwargs["vertexai"] is True
    assert client.init_kwargs["project"] == "my-project"
    assert client.init_kwargs["location"] == "us-central1"
    # The request used the configured model and forwarded the prompt + config.
    (call,) = client.calls
    assert call["model"] == "gemini-custom"
    assert call["contents"] == "hello"
    cfg = call["config"]
    assert isinstance(cfg, _FakeGenerateContentConfig)
    assert cfg.kwargs["system_instruction"] == "be brief"
    assert cfg.kwargs["max_output_tokens"] == 64
    assert cfg.kwargs["temperature"] == 0.0


def test_complete_without_system(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    adapter = VertexLLM(project="p", location="l")
    adapter.complete("just user")
    (call,) = adapter._ensure_client().calls
    assert call["contents"] == "just user"
    assert call["config"].kwargs["system_instruction"] is None


def test_none_text_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = _install_fake_genai(monkeypatch)
    monkeypatch.setattr(fake_cls, "reply", None)
    adapter = VertexLLM(project="p", location="l")
    assert adapter.complete("x") == ""


def test_reads_project_location_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")

    adapter = VertexLLM()
    assert adapter.complete("hi") == "fake-vertex-response"
    client = adapter._ensure_client()
    assert client.init_kwargs["project"] == "env-project"
    assert client.init_kwargs["location"] == "europe-west4"


def test_vendor_failure_wrapped_in_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)
    adapter = VertexLLM(project="p", location="l")

    def _boom(self: _FakeModels, **kwargs: Any) -> _FakeResponse:
        raise RuntimeError("vertex down")

    monkeypatch.setattr(_FakeModels, "generate_content", _boom)
    with pytest.raises(StageError):
        adapter.complete("x")


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction must fail with MissingExtraError.

    require_extra is NOT neutralized here, so the absent ``gcp`` extra surfaces.
    """
    with pytest.raises(MissingExtraError):
        VertexLLM(project="p", location="l")
