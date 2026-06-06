"""Unit tests for :mod:`indx.llm.litellm` with the vendor ``litellm`` SDK mocked.

The ``litellm`` extra is absent in the offline test environment. These tests inject a fake
``litellm`` module exposing ``completion`` and neutralize ``require_extra`` so the adapter
constructs, then assert it satisfies the ``LLM`` protocol and round-trips core types
(``str`` in, ``str`` out). One test runs in the real, dep-absent environment and asserts that
construction raises :class:`MissingExtraError`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.llm.litellm  # noqa: F401  -- force-load before any require_extra patch
from indx.errors import MissingExtraError
from indx.llm.base import LLM
from indx.llm.litellm import LiteLLMClient


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.litellm.require_extra", lambda *a, **k: None)
    module = types.ModuleType("litellm")
    module.last_request = {}  # type: ignore[attr-defined]

    def completion(**kwargs: Any) -> _Completion:
        module.last_request = kwargs  # type: ignore[attr-defined]
        return _Completion("remote-first; work from anywhere")

    module.completion = completion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def test_satisfies_llm_protocol(fake_litellm: types.ModuleType) -> None:
    assert isinstance(LiteLLMClient(), LLM)


def test_default_model(fake_litellm: types.ModuleType) -> None:
    assert LiteLLMClient().model == "gpt-4o-mini"


def test_provider_model_string_is_passed_through(fake_litellm: types.ModuleType) -> None:
    client = LiteLLMClient(model="bedrock/anthropic.claude-3")
    out = client.complete("summarize", system="be terse", max_tokens=64, temperature=0.0)
    assert out == "remote-first; work from anywhere"
    req = fake_litellm.last_request
    assert req["model"] == "bedrock/anthropic.claude-3"
    assert req["max_tokens"] == 64
    # drop_params lets LiteLLM discard provider-unsupported params (e.g. temperature).
    assert req["drop_params"] is True
    assert req["messages"][0] == {"role": "system", "content": "be terse"}
    assert req["messages"][1] == {"role": "user", "content": "summarize"}


def test_api_base_from_env(fake_litellm: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:11434")
    LiteLLMClient(model="ollama/llama3").complete("hi")
    assert fake_litellm.last_request["api_base"] == "http://localhost:11434"


def test_none_content_becomes_empty_string(
    fake_litellm: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fake_litellm, "completion", lambda **k: _Completion(None))
    assert LiteLLMClient().complete("x") == ""


def test_construction_without_extra_raises_missing_extra() -> None:
    with pytest.raises(MissingExtraError):
        LiteLLMClient()
