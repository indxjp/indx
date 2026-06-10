"""Mocked unit tests for :class:`indx.llm.azure.AzureOpenAILLM`.

The ``openai`` SDK is not installed in this environment, so it is faked: a stand-in
``openai`` module is injected into ``sys.modules`` and the dependency gate is neutralized.
Tests are deterministic and offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.llm.azure import AzureOpenAILLM
from indx.llm.base import LLM


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

    reply: str | None = "fake-azure-response"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAzureOpenAI]:
    """Neutralize the extra gate and inject a fake ``openai`` module."""
    # azure.py binds require_extra at import time (``from ... import require_extra``),
    # so patch the name in both the source module and the consuming module.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.azure.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("openai")
    fake_module.AzureOpenAI = _FakeAzureOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return _FakeAzureOpenAI


def test_satisfies_llm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAILLM(
        model="my-deployment",
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
    )
    assert adapter.name == "azure"
    assert isinstance(adapter, LLM)


def test_complete_round_trips_through_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAILLM(
        model="my-deployment",
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
    )

    out = adapter.complete("hello", system="be brief", max_tokens=64, temperature=0.0)

    assert out == "fake-azure-response"
    client = adapter._ensure_client()  # cached fake
    assert isinstance(client, _FakeAzureOpenAI)
    # Client constructed with Azure-specific config.
    assert client.init_kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert client.init_kwargs["api_key"] == "secret-key"
    assert client.init_kwargs["api_version"] == "2024-10-21"
    # The request used the deployment as the model and forwarded both messages. The token cap
    # is sent as ``max_completion_tokens`` (gpt-5/o-series reject the legacy ``max_tokens``);
    # a non-reasoning deployment still forwards a custom temperature.
    (call,) = client.calls
    assert call["model"] == "my-deployment"
    assert call["max_completion_tokens"] == 64
    assert "max_tokens" not in call
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


class _FakeBadRequest(Exception):
    """Stand-in for ``openai.BadRequestError`` (matched by class name + status_code)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = 400


_FakeBadRequest.__name__ = "BadRequestError"


def test_create_retries_when_deployment_rejects_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque (mis-guessed non-reasoning) deployment that rejects temperature recovers.

    The first attempt sends ``temperature``; the model 400s ("does not support"); the adapter
    drops temperature, adds ``reasoning_effort=minimal``, and retries to success — no 400 leaks.
    """
    _install_fake_openai(monkeypatch)
    # A neutral deployment name → first attempt includes temperature.
    adapter = AzureOpenAILLM(
        model="prod-llm",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
        api_version="2024-10-21",
    )
    client = adapter._ensure_client()
    real_create = client.chat.completions.create

    def flaky_create(**kwargs: Any) -> Any:
        if "temperature" in kwargs:
            raise _FakeBadRequest(
                "Unsupported value: 'temperature' does not support 0.0 with this model."
            )
        return real_create(**kwargs)

    monkeypatch.setattr(client.chat.completions, "create", flaky_create)

    out = adapter.complete("hi", max_tokens=64, temperature=0.0)
    assert out == "fake-azure-response"
    # The recorded (successful) call dropped temperature and added reasoning_effort.
    (call,) = client.calls
    assert "temperature" not in call
    assert call["reasoning_effort"] == "minimal"
    assert call["max_completion_tokens"] == 64


def test_create_recovers_from_unrecognized_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment mis-guessed as reasoning that 400s on ``reasoning_effort`` recovers.

    The 400 wording — "Unrecognized request argument supplied: reasoning_effort" — contains
    "unrecognized"/"supplied" but not "support"; the adapter must still recognise it, drop
    ``reasoning_effort``, swap in ``temperature`` and retry to success.
    """
    _install_fake_openai(monkeypatch)
    # A gpt-5 name → first attempt sends reasoning_effort (no temperature).
    adapter = AzureOpenAILLM(
        model="gpt-5-prod",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
        api_version="2024-10-21",
    )
    client = adapter._ensure_client()
    real_create = client.chat.completions.create

    def flaky_create(**kwargs: Any) -> Any:
        if "reasoning_effort" in kwargs:
            raise _FakeBadRequest("Unrecognized request argument supplied: reasoning_effort")
        return real_create(**kwargs)

    monkeypatch.setattr(client.chat.completions, "create", flaky_create)

    out = adapter.complete("hi", max_tokens=64, temperature=0.0)
    assert out == "fake-azure-response"
    (call,) = client.calls
    assert "reasoning_effort" not in call
    assert call["temperature"] == 0.0
    assert call["max_completion_tokens"] == 64


def test_reasoning_deployment_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gpt-5/o-series deployment must omit ``temperature`` (the model rejects a custom one)."""
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAILLM(
        model="gpt-5-mini-prod",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
        api_version="2024-10-21",
    )
    adapter.complete("hi", max_tokens=16, temperature=0.0)
    (call,) = adapter._ensure_client().calls  # type: ignore[attr-defined]
    assert call["max_completion_tokens"] == 16
    assert "max_tokens" not in call
    assert "temperature" not in call  # omitted for reasoning models


@pytest.mark.parametrize("deployment", ["gpt-5-chat", "gpt-5-chat-latest-prod"])
def test_gpt5_chat_deployment_forwards_temperature(
    monkeypatch: pytest.MonkeyPatch, deployment: str
) -> None:
    """A gpt-5-chat* deployment is a non-reasoning model: keep ``temperature``, no effort knob."""
    assert AzureOpenAILLM._is_reasoning_model(deployment) is False

    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAILLM(
        model=deployment,
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
        api_version="2024-10-21",
    )
    adapter.complete("hi", max_tokens=16, temperature=0.3)
    (call,) = adapter._ensure_client().calls  # type: ignore[attr-defined]
    assert call["max_completion_tokens"] == 16
    assert call["temperature"] == 0.3
    assert "reasoning_effort" not in call
    assert "max_tokens" not in call


def test_complete_without_system_omits_system_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    adapter = AzureOpenAILLM(
        model="dep",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
    )
    adapter.complete("just user")
    (call,) = adapter._ensure_client().calls
    assert call["messages"] == [{"role": "user", "content": "just user"}]


def test_none_content_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = _install_fake_openai(monkeypatch)
    monkeypatch.setattr(fake_cls, "reply", None)
    adapter = AzureOpenAILLM(
        model="dep",
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
    )
    assert adapter.complete("x") == ""


def test_reads_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")

    adapter = AzureOpenAILLM()
    assert adapter.complete("hi") == "fake-azure-response"
    client = adapter._ensure_client()
    assert client.init_kwargs["api_key"] == "env-key"
    assert client.init_kwargs["azure_endpoint"] == "https://env.openai.azure.com"
    assert client.calls[0]["model"] == "env-deployment"


def test_missing_deployment_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    adapter = AzureOpenAILLM(
        api_key="k",
        azure_endpoint="https://e.openai.azure.com",
    )
    with pytest.raises(StageError):
        adapter.complete("x")


def test_missing_api_key_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    adapter = AzureOpenAILLM(
        model="dep",
        azure_endpoint="https://e.openai.azure.com",
    )
    with pytest.raises(StageError):
        adapter.complete("x")


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction must fail with MissingExtraError.

    require_extra is NOT neutralized here, so the absent ``openai`` extra surfaces.
    """
    with pytest.raises(MissingExtraError):
        AzureOpenAILLM(model="dep", api_key="k", azure_endpoint="https://e.openai.azure.com")
