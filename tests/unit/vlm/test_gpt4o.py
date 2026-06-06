"""Mocked unit tests for GPT4oVLM — the openai SDK is absent, so it is faked.

The vendor ``openai`` module is injected into ``sys.modules`` and the dependency gate is
neutralized so construction proceeds offline (coding-standards §11). The real dep-absent
construction is asserted in a separate test that does *not* neutralize the gate.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError
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
    def __init__(self, client: _FakeOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self._client.last_kwargs = kwargs
        return _FakeCompletion(self._client.reply)


class _FakeChat:
    def __init__(self, client: _FakeOpenAI) -> None:
        self.completions = _FakeCompletions(client)


class _FakeOpenAI:
    reply: str | None = "a red square on a white background"

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.last_kwargs: dict[str, Any] = {}
        self.chat = _FakeChat(self)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenAI]:
    """Inject a fake ``openai`` module and neutralize the require_extra gate.

    The gate is neutralized on the adapter module itself. ``indx.vlm.gpt4o`` binds
    ``require_extra`` by name at import time, so we import it first (before patching) to
    capture the real reference, then patch the bound name; monkeypatch restores it on
    teardown so the dep-absent test still sees the real gate.
    """
    import indx.vlm.gpt4o as gpt4o_mod

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setattr(gpt4o_mod, "require_extra", lambda *a, **k: None)
    return _FakeOpenAI


def test_satisfies_vlm_protocol(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.vlm.gpt4o import GPT4oVLM

    assert isinstance(GPT4oVLM(), VLM)
    assert GPT4oVLM().name == "gpt4o"


def test_default_model_is_gpt4o(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.vlm.gpt4o import GPT4oVLM

    assert GPT4oVLM().model == "gpt-4o"
    assert GPT4oVLM("gpt-4o-mini").model == "gpt-4o-mini"


def test_describe_returns_caption_text(fake_openai: type[_FakeOpenAI]) -> None:
    from indx.vlm.gpt4o import GPT4oVLM

    vlm = GPT4oVLM()
    out = vlm.describe(b"\x89PNG fake bytes", prompt="What is this?")

    assert out == "a red square on a white background"


def test_describe_builds_base64_data_url_and_is_deterministic(
    fake_openai: type[_FakeOpenAI],
) -> None:
    from indx.vlm.gpt4o import GPT4oVLM

    vlm = GPT4oVLM()
    image = b"\x89PNG fake bytes"
    vlm.describe(image, prompt="caption please")

    client = vlm._ensure_client()
    kwargs = client.last_kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.0  # determinism

    content = kwargs["messages"][0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image_url")
    assert text_part["text"] == "caption please"
    expected_b64 = base64.b64encode(image).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


def test_describe_handles_empty_content(
    monkeypatch: pytest.MonkeyPatch, fake_openai: type[_FakeOpenAI]
) -> None:
    from indx.vlm.gpt4o import GPT4oVLM

    monkeypatch.setattr(_FakeOpenAI, "reply", None)
    assert GPT4oVLM().describe(b"x") == ""


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.vlm.gpt4o import GPT4oVLM

    # Guard against fake-module leakage from other tests: the real env has no openai.
    assert "openai" not in sys.modules
    assert importlib.util.find_spec("openai") is None

    with pytest.raises(MissingExtraError) as exc_info:
        GPT4oVLM()
    assert exc_info.value.extra == "openai"
