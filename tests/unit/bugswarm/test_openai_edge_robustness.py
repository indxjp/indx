"""Regression tests for :mod:`indx.llm.openai` edge-conversion robustness.

These cover two confirmed bugswarm findings for the OpenAI adapter:

* an empty ``choices`` array (a real provider edge case, e.g. an Azure/OpenAI-compatible
  content-filter block) must yield ``""`` instead of raising ``IndexError``; and
* a vendor SDK failure from ``chat.completions.create`` must be normalized to a typed
  :class:`~indx.errors.StageError` at the adapter edge, matching the bedrock/ollama/vertex
  adapters in the same slot.

The ``openai`` extra is intentionally absent offline, so a fake ``openai`` module is injected
into ``sys.modules`` and ``require_extra`` is neutralized, mirroring ``tests/unit/llm``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import StageError
from indx.llm.openai import OpenAILLM


class _FakeEmptyCompletion:
    """A vendor response with zero choices (content-filter / error edge case)."""

    def __init__(self) -> None:
        self.choices: list[Any] = []


class _FakeCompletionsEmpty:
    def create(self, **kwargs: Any) -> _FakeEmptyCompletion:
        return _FakeEmptyCompletion()


class _FakeCompletionsRaising:
    def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("simulated vendor failure (rate limit / auth / 400)")


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _FakeOpenAI:
    """Stand-in for ``openai.OpenAI`` whose completions behavior is injected per-test."""

    _completions: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.chat = _FakeChat(type(self)._completions)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, completions: Any) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.openai.require_extra", lambda *a, **k: None)

    client_cls = type("_FakeOpenAIClient", (_FakeOpenAI,), {"_completions": completions})
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = client_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_complete_returns_empty_string_for_zero_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, _FakeCompletionsEmpty())
    adapter = OpenAILLM()

    result = adapter.complete("Summarize this.")

    assert result == ""


def test_complete_wraps_vendor_error_as_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, _FakeCompletionsRaising())
    adapter = OpenAILLM()

    with pytest.raises(StageError) as excinfo:
        adapter.complete("Summarize this.")

    assert excinfo.value.stage == "enrich"
    # The original vendor exception is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
