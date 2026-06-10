"""Unit tests for :mod:`indx.llm.openai` with the vendor ``openai`` SDK mocked.

The ``openai`` extra is intentionally absent in the offline test environment. These tests
inject a fake ``openai`` module into ``sys.modules`` and neutralize ``require_extra`` so the
adapter constructs, then assert it satisfies the ``LLM`` protocol and round-trips core types
(``str`` in, ``str`` out). One test runs in the real, dep-absent environment and asserts that
construction raises :class:`MissingExtraError`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError
from indx.llm.base import LLM
from indx.llm.openai import OpenAILLM


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
    def __init__(self, parent: _FakeOpenAI) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _FakeCompletion:
        # Record the exact request so tests can assert determinism defaults.
        self._parent.last_request = kwargs
        return _FakeCompletion(self._parent.reply)


class _FakeChat:
    def __init__(self, parent: _FakeOpenAI) -> None:
        self.completions = _FakeCompletions(parent)


class _FakeOpenAI:
    """Stand-in for ``openai.OpenAI`` capturing init args and serving a canned reply."""

    reply: str | None = "fake-completion-text"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.last_request: dict[str, Any] | None = None
        self.chat = _FakeChat(self)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the extra gate and inject a fake ``openai`` module."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter does ``from indx.utils.lazy import require_extra``, binding the name in its
    # own module namespace, so patch that binding too for the gate to be fully neutralized.
    monkeypatch.setattr("indx.llm.openai.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_satisfies_llm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM()
    assert isinstance(adapter, LLM)
    assert adapter.name == "openai"
    assert adapter.model == "gpt-5-mini"


def test_custom_model_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM(model="gpt-4o-mini")
    assert adapter.model == "gpt-4o-mini"


def test_complete_returns_message_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM()

    result = adapter.complete("Summarize this.")

    assert isinstance(result, str)
    assert result == "fake-completion-text"


def test_complete_defaults_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM()

    adapter.complete("hello")

    client = adapter._ensure_client()
    request = client.last_request  # type: ignore[attr-defined]
    assert request is not None
    assert request["model"] == "gpt-5-mini"
    # The default gpt-5 family model rejects ``max_tokens`` and custom ``temperature`` on
    # Chat Completions, so the adapter sends ``max_completion_tokens`` and omits temperature.
    assert request["max_completion_tokens"] == 512
    assert "max_tokens" not in request
    assert "temperature" not in request
    # Reasoning models get reasoning_effort=minimal so a tight token budget still yields content.
    assert request["reasoning_effort"] == "minimal"
    assert request["messages"] == [{"role": "user", "content": "hello"}]


def test_complete_omits_temperature_for_reasoning_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM()

    # Even when a caller passes an explicit temperature, the gpt-5 family default must not
    # receive it (the model returns HTTP 400 for any non-default temperature).
    adapter.complete("body", system="you are terse", max_tokens=64, temperature=0.7)

    client = adapter._ensure_client()
    request = client.last_request  # type: ignore[attr-defined]
    assert request is not None
    assert request["messages"] == [
        {"role": "system", "content": "you are terse"},
        {"role": "user", "content": "body"},
    ]
    assert request["max_completion_tokens"] == 64
    assert "max_tokens" not in request
    assert "temperature" not in request


def test_complete_forwards_temperature_for_non_reasoning_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM(model="gpt-4o-mini")

    adapter.complete("body", max_tokens=64, temperature=0.7)

    client = adapter._ensure_client()
    request = client.last_request  # type: ignore[attr-defined]
    assert request is not None
    assert request["max_completion_tokens"] == 64
    assert request["temperature"] == 0.7
    assert "max_tokens" not in request
    # Non-reasoning models keep their custom temperature and get no reasoning_effort knob.
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("model", ["gpt-5-chat", "gpt-5-chat-latest"])
def test_complete_treats_gpt5_chat_as_non_reasoning(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    # gpt-5-chat* are ordinary chat models: they accept a custom temperature and REJECT
    # reasoning_effort. The startswith('gpt-5') prefix would otherwise misclassify them as
    # reasoning models, so they must be carved out (see OpenAILLM._is_reasoning_model).
    assert OpenAILLM._is_reasoning_model(model) is False

    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM(model=model)

    adapter.complete("body", max_tokens=64, temperature=0.7)

    client = adapter._ensure_client()
    request = client.last_request  # type: ignore[attr-defined]
    assert request is not None
    assert request["max_completion_tokens"] == 64
    # Custom temperature is forwarded and the rejected reasoning_effort knob is never sent.
    assert request["temperature"] == 0.7
    assert "reasoning_effort" not in request
    assert "max_tokens" not in request


def test_complete_handles_none_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    sys.modules["openai"].OpenAI.reply = None  # type: ignore[attr-defined]
    try:
        adapter = OpenAILLM()
        assert adapter.complete("anything") == ""
    finally:
        sys.modules["openai"].OpenAI.reply = "fake-completion-text"  # type: ignore[attr-defined]


def test_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    _install_fake_openai(monkeypatch)
    adapter = OpenAILLM()

    client = adapter._ensure_client()
    assert client.init_kwargs["api_key"] == "sk-from-env"  # type: ignore[attr-defined]


def test_missing_extra_raises_in_dep_absent_env() -> None:
    # No neutralization of require_extra: the real (absent) ``openai`` extra must fail fast.
    with pytest.raises(MissingExtraError) as excinfo:
        OpenAILLM()
    assert excinfo.value.extra == "openai"
    assert "pip install indx[openai]" in str(excinfo.value)
