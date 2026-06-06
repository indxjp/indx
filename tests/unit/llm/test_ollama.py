"""Unit tests for :class:`indx.llm.ollama.OllamaLLM`.

The ``ollama`` vendor SDK is absent in the offline test environment, so it is faked: a
stub module is injected into ``sys.modules`` and the dependency gate is neutralised so
construction proceeds. Tests stay deterministic and offline (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.llm.base import LLM


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeClient:
    """Records the last chat() call and returns a canned response."""

    reply: str | None = "the fake reply"
    raise_on_chat: bool = False

    def __init__(self, host: str | None = None) -> None:
        self.host = host
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self.raise_on_chat:
            raise RuntimeError("connection refused")
        return _FakeResponse(self.reply)


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Neutralise the extra gate and inject a fake ``ollama`` module."""
    # Import the adapter so its bound ``require_extra`` name exists to patch.
    import indx.llm.ollama  # noqa: F401

    # Neutralise the gate at both the source and the adapter's bound name (the adapter
    # imports require_extra by name, so patching only the source module is not enough).
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.ollama.require_extra", lambda *a, **k: None)
    fake = types.ModuleType("ollama")
    fake.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake)
    return _FakeClient


def test_construction_without_extra_raises() -> None:
    """In the real dep-absent environment construction raises MissingExtraError."""
    from indx.llm.ollama import OllamaLLM

    with pytest.raises(MissingExtraError) as exc_info:
        OllamaLLM()
    assert exc_info.value.extra == "ollama"


def test_satisfies_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ollama(monkeypatch)
    from indx.llm.ollama import OllamaLLM

    adapter = OllamaLLM()
    assert isinstance(adapter, LLM)
    assert adapter.name == "ollama"


def test_default_model_is_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ollama(monkeypatch)
    from indx.llm.ollama import OllamaLLM

    assert OllamaLLM().model == "qwen2.5"
    assert OllamaLLM(model="llama3").model == "llama3"


def test_complete_round_trips_to_str(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ollama(monkeypatch)
    from indx.llm.ollama import OllamaLLM

    adapter = OllamaLLM(model="qwen2.5")
    out = adapter.complete("hello")

    assert out == "the fake reply"
    assert isinstance(out, str)

    client = adapter._ensure_client()
    call = client.calls[-1]
    assert call["model"] == "qwen2.5"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    # Determinism: temperature defaults to 0.0; max_tokens maps to num_predict.
    assert call["options"]["temperature"] == 0.0
    assert call["options"]["num_predict"] == 512


def test_complete_includes_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ollama(monkeypatch)
    from indx.llm.ollama import OllamaLLM

    adapter = OllamaLLM()
    adapter.complete("q", system="be terse", max_tokens=64, temperature=0.7)

    call = adapter._ensure_client().calls[-1]
    assert call["messages"][0] == {"role": "system", "content": "be terse"}
    assert call["messages"][1] == {"role": "user", "content": "q"}
    assert call["options"]["temperature"] == 0.7
    assert call["options"]["num_predict"] == 64


def test_complete_empty_content_returns_empty_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls = _install_fake_ollama(monkeypatch)
    monkeypatch.setattr(fake_cls, "reply", None)
    from indx.llm.ollama import OllamaLLM

    assert OllamaLLM().complete("x") == ""


def test_host_passed_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ollama(monkeypatch)
    from indx.llm.ollama import OllamaLLM

    adapter = OllamaLLM(host="http://localhost:11434")
    client = adapter._ensure_client()
    assert client.host == "http://localhost:11434"


def test_chat_failure_raises_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cls = _install_fake_ollama(monkeypatch)
    monkeypatch.setattr(fake_cls, "raise_on_chat", True)
    from indx.llm.ollama import OllamaLLM

    with pytest.raises(StageError) as exc_info:
        OllamaLLM().complete("x")
    assert exc_info.value.stage == "enrich"
