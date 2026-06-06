"""Mocked unit tests for :class:`indx.vlm.azure.AzureOpenAIVLM`.

The ``openai`` SDK is not installed in this environment, so it is faked: a stand-in
``openai`` module is injected into ``sys.modules`` and the dependency gate is neutralized.
Tests are deterministic and offline. The real dep-absent construction is asserted in a
separate test that does *not* neutralize the gate.
"""

from __future__ import annotations

import base64
import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.vlm.azure import AzureOpenAIVLM
from indx.vlm.base import VLM


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, client: _FakeAzureOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self._client.calls.append(kwargs)
        return _FakeCompletion(self._client.reply)


class _FakeChat:
    def __init__(self, client: _FakeAzureOpenAI) -> None:
        self.completions = _FakeCompletions(client)


class _FakeAzureOpenAI:
    """Minimal stand-in for ``openai.AzureOpenAI`` that records call kwargs."""

    reply: str | None = "a red square on a white background"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAzureOpenAI]:
    """Neutralize the extra gate and inject a fake ``openai`` module."""
    # azure.py binds require_extra at import time (``from ... import require_extra``),
    # so patch the name in both the source module and the consuming module.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.vlm.azure.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("openai")
    fake_module.AzureOpenAI = _FakeAzureOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return _FakeAzureOpenAI


def test_satisfies_vlm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAIVLM(
        deployment="my-vlm-deployment",
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
    )
    assert adapter.name == "azure"
    assert isinstance(adapter, VLM)


def test_describe_returns_caption_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAIVLM(
        deployment="my-vlm-deployment",
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
    )

    out = adapter.describe(b"\x89PNG fake bytes", prompt="What is this?")

    assert out == "a red square on a white background"


def test_describe_builds_base64_data_url_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAIVLM(
        deployment="my-vlm-deployment",
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
    )
    image = b"\x89PNG fake bytes"
    adapter.describe(image, prompt="caption please")

    client = adapter._ensure_client()  # cached fake
    assert isinstance(client, _FakeAzureOpenAI)
    # Client constructed with Azure-specific config.
    assert client.init_kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert client.init_kwargs["api_key"] == "secret-key"
    assert client.init_kwargs["api_version"] == "2024-10-21"

    (call,) = client.calls
    assert call["model"] == "my-vlm-deployment"  # deployment used as the model
    assert call["temperature"] == 0.0  # determinism
    content = call["messages"][0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image_url")
    assert text_part["text"] == "caption please"
    expected_b64 = base64.b64encode(image).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


def test_describe_reads_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_VLM_DEPLOYMENT", "env-vlm-deployment")

    adapter = AzureOpenAIVLM()
    assert adapter.describe(b"x") == "a red square on a white background"
    client = adapter._ensure_client()
    assert client.init_kwargs["api_key"] == "env-key"
    assert client.init_kwargs["azure_endpoint"] == "https://env.openai.azure.com"
    assert client.calls[0]["model"] == "env-vlm-deployment"


def test_describe_handles_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = _install_fake_openai(monkeypatch)
    monkeypatch.setattr(fake_cls, "reply", None)
    adapter = AzureOpenAIVLM(
        deployment="dep",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
    )
    assert adapter.describe(b"x") == ""


def test_missing_deployment_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_VLM_DEPLOYMENT", raising=False)
    adapter = AzureOpenAIVLM(
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
    )
    with pytest.raises(StageError):
        adapter.describe(b"x")


def test_missing_api_key_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    adapter = AzureOpenAIVLM(
        deployment="dep",
        azure_endpoint="https://e.openai.azure.com",
    )
    with pytest.raises(StageError):
        adapter.describe(b"x")


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction must fail with MissingExtraError.

    require_extra is NOT neutralized here, so the absent ``openai`` extra surfaces.
    """
    with pytest.raises(MissingExtraError) as exc_info:
        AzureOpenAIVLM(
            deployment="dep",
            api_key="k",
            azure_endpoint="https://e.openai.azure.com",
        )
    assert exc_info.value.extra == "azure"
