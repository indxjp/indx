"""Mocked unit tests for VLLMClient — the openai SDK is absent, so it is faked.

vLLM speaks the OpenAI-compatible HTTP API, so the adapter uses the ``openai`` client. The
vendor ``openai`` module is injected into ``sys.modules`` and the dependency gate is
neutralized so construction proceeds offline (coding-standards §11). The real dep-absent
construction is asserted in a separate test that does *not* neutralize the gate.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError
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
    def __init__(self, client: _FakeOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self._client.last_kwargs = kwargs
        return _FakeCompletion(self._client.reply)


class _FakeChat:
    def __init__(self, client: _FakeOpenAI) -> None:
        self.completions = _FakeCompletions(client)


class _FakeOpenAI:
    reply: str | None = "the quick brown fox"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.last_kwargs: dict[str, Any] = {}
        self.chat = _FakeChat(self)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenAI]:
    """Inject a fake ``openai`` module and neutralize the require_extra gate."""
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    # Patch the *binding used by the adapter*: vllm.py does ``from indx.utils.lazy import
    # require_extra``, so patching ``indx.utils.lazy.require_extra`` would miss the local name.
    monkeypatch.setattr("indx.llm.vllm.require_extra", lambda *a, **k: None)
    return _FakeOpenAI


def test_satisfies_llm_protocol(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.llm.vllm import VLLMClient

    assert isinstance(VLLMClient(), LLM)
    assert VLLMClient().name == "vllm"


def test_defaults(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.llm.vllm import VLLMClient

    client = VLLMClient()
    assert client.model == "qwen2.5"
    assert client._base_url == "http://localhost:8000/v1"
    # No OPENAI_API_KEY set in the test env -> the throwaway default.
    assert client._api_key == "EMPTY"


def test_model_and_base_url_overrides(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.llm.vllm import VLLMClient

    client = VLLMClient("mistral-7b", base_url="http://gpu-box:9000/v1")
    assert client.model == "mistral-7b"
    assert client._base_url == "http://gpu-box:9000/v1"


def test_api_key_read_from_env(
    monkeypatch: pytest.MonkeyPatch, fake_openai: type[_FakeOpenAI]
) -> None:
    from indx.llm.vllm import VLLMClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert VLLMClient()._api_key == "sk-from-env"


def test_complete_returns_text(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.llm.vllm import VLLMClient

    out = VLLMClient().complete("hello")
    assert out == "the quick brown fox"


def test_complete_passes_deterministic_request(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.llm.vllm import VLLMClient

    llm = VLLMClient("qwen2.5")
    llm.complete("summarize this", system="be terse", max_tokens=128, temperature=0.0)
    client = llm._ensure_client()
    kwargs = client.last_kwargs
    assert kwargs["model"] == "qwen2.5"
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.0  # determinism (§1.4)
    assert client.base_url == "http://localhost:8000/v1"
    assert kwargs["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "summarize this"},
    ]


def test_complete_without_system_omits_system_message(
    fake_openai: type[_FakeOpenAI],
) -> None:
    from indx.llm.vllm import VLLMClient

    llm = VLLMClient()
    llm.complete("just the user turn")
    kwargs = llm._ensure_client().last_kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "just the user turn"}]


def test_complete_handles_empty_content(
    monkeypatch: pytest.MonkeyPatch, fake_openai: type[_FakeOpenAI]
) -> None:
    from indx.llm.vllm import VLLMClient

    monkeypatch.setattr(_FakeOpenAI, "reply", None)
    assert VLLMClient().complete("anything") == ""


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.llm.vllm import VLLMClient

    with pytest.raises(MissingExtraError) as exc_info:
        VLLMClient()
    assert exc_info.value.extra == "vllm"
