"""Mocked unit tests for AnthropicLLM — the anthropic SDK is absent, so it is faked.

The vendor ``anthropic`` module is injected into ``sys.modules`` and the dependency gate is
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


class _FakeBlock:
    """A single content block, mimicking anthropic's typed blocks (``.type`` + ``.text``)."""

    def __init__(self, block_type: str, text: str = "") -> None:
        self.type = block_type
        self.text = text


class _FakeMessage:
    def __init__(self, content: list[_FakeBlock]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, client: _FakeAnthropic) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _FakeMessage:
        self._client.last_kwargs = kwargs
        return _FakeMessage(self._client.blocks)


class _FakeAnthropic:
    # Default reply: a thinking block (skipped) plus two text blocks (concatenated).
    blocks: list[_FakeBlock] = [
        _FakeBlock("thinking", "internal reasoning that must not leak"),
        _FakeBlock("text", "summary: "),
        _FakeBlock("text", "a concise enrichment"),
    ]

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.last_kwargs: dict[str, Any] = {}
        self.messages = _FakeMessages(self)


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAnthropic]:
    """Inject a fake ``anthropic`` module and neutralize the require_extra gate."""
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    # Import the adapter here so its module exists, then patch the require_extra reference *in its
    # own namespace* (the adapter does ``from indx.utils.lazy import require_extra``, so patching
    # indx.utils.lazy would not affect the already-bound name). monkeypatch restores it cleanly,
    # so the later dep-absent test sees the real gate.
    import indx.llm.anthropic as adapter

    monkeypatch.setattr(adapter, "require_extra", lambda *a, **k: None)
    return _FakeAnthropic


def test_satisfies_llm_protocol(fake_anthropic: type[_FakeAnthropic]) -> None:
    from indx.llm.anthropic import AnthropicLLM

    assert isinstance(AnthropicLLM(), LLM)
    assert AnthropicLLM().name == "anthropic"


def test_default_model_and_override(fake_anthropic: type[_FakeAnthropic]) -> None:
    from indx.llm.anthropic import AnthropicLLM

    assert AnthropicLLM().model == "claude-haiku-4-5"
    assert AnthropicLLM("claude-opus-4-8").model == "claude-opus-4-8"


def test_complete_concatenates_text_blocks_only(
    fake_anthropic: type[_FakeAnthropic],
) -> None:
    from indx.llm.anthropic import AnthropicLLM

    out = AnthropicLLM().complete("Summarize this chunk.")

    # Only the two text blocks are joined; the thinking block is skipped.
    assert out == "summary: a concise enrichment"


def test_complete_maps_system_max_tokens_temperature(
    fake_anthropic: type[_FakeAnthropic],
) -> None:
    from indx.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM()
    llm.complete("hello", system="You are terse.", max_tokens=128, temperature=0.0)

    kwargs = llm._ensure_client().last_kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.0  # deterministic default
    assert kwargs["system"] == "You are terse."
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_complete_omits_system_when_none(
    fake_anthropic: type[_FakeAnthropic],
) -> None:
    from indx.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM()
    llm.complete("hello")

    # The Messages API rejects system=None; an absent key means "no system prompt".
    assert "system" not in llm._ensure_client().last_kwargs


def test_complete_handles_no_text_blocks(
    monkeypatch: pytest.MonkeyPatch, fake_anthropic: type[_FakeAnthropic]
) -> None:
    from indx.llm.anthropic import AnthropicLLM

    monkeypatch.setattr(_FakeAnthropic, "blocks", [_FakeBlock("thinking", "no text here")])
    assert AnthropicLLM().complete("x") == ""


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.llm.anthropic import AnthropicLLM

    with pytest.raises(MissingExtraError) as exc_info:
        AnthropicLLM()
    assert exc_info.value.extra == "anthropic"
