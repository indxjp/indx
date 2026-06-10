"""Regression tests for :class:`indx.vlm.azure.AzureOpenAIVLM` bug-swarm fixes.

Covers four confirmed defects that were fixed by porting sibling-adapter behavior:

1. The data: URL hard-coded ``image/png``, rejecting valid JPEG/GIF/WebP figures.
2. ``choices[0]`` was indexed without guarding an empty (content-filtered) choices list.
3. ``temperature=0.0`` was sent unconditionally, breaking gpt-5/o-series reasoning deployments.
4. The vendor ``create`` call was unwrapped, leaking raw SDK exceptions past the adapter edge.

The ``openai`` SDK is faked and injected into ``sys.modules`` so the suite stays offline.
"""

from __future__ import annotations

import base64
import sys
import types
from typing import Any

import pytest

from indx.errors import StageError
from indx.vlm.azure import AzureOpenAIVLM


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


class _FakeCompletions:
    def __init__(self, client: _FakeAzureOpenAI) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self._client.calls.append(kwargs)
        if self._client.error is not None:
            err = self._client.error
            # Allow a one-shot error to drive the resilient-retry path.
            if self._client.error_once:
                self._client.error = None
            raise err
        return _FakeCompletion(self._client.choices)


class _FakeChat:
    def __init__(self, client: _FakeAzureOpenAI) -> None:
        self.completions = _FakeCompletions(client)


class _FakeAzureOpenAI:
    """Minimal stand-in for ``openai.AzureOpenAI`` recording calls and optional errors."""

    choices: list[_FakeChoice] = [_FakeChoice("a red square")]
    error: Exception | None = None
    error_once: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


class _FakeBadRequestError(Exception):
    """Mimics ``openai.BadRequestError`` shape used by ``_unsupported_param``."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = 400


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAzureOpenAI]:
    monkeypatch.setattr("indx.vlm.azure.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("openai")
    fake_module.AzureOpenAI = _FakeAzureOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    # Reset class-level state so tests do not bleed into one another.
    _FakeAzureOpenAI.choices = [_FakeChoice("a red square")]
    _FakeAzureOpenAI.error = None
    _FakeAzureOpenAI.error_once = False
    return _FakeAzureOpenAI


def _make_adapter(deployment: str = "my-vlm-deployment") -> AzureOpenAIVLM:
    return AzureOpenAIVLM(
        deployment=deployment,
        api_key="secret-key",
        azure_endpoint="https://example.openai.azure.com",
    )


def test_jpeg_bytes_get_jpeg_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JPEG figure must be declared image/jpeg, not the old hard-coded image/png."""
    client_cls = _install_fake_openai(monkeypatch)
    adapter = _make_adapter()
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF fake jpeg bytes"

    adapter.describe(jpeg)

    client = adapter._ensure_client()
    assert isinstance(client, client_cls)
    (call,) = client.calls
    image_part = next(p for p in call["messages"][0]["content"] if p["type"] == "image_url")
    expected_b64 = base64.b64encode(jpeg).decode("ascii")
    assert image_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


def test_empty_choices_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """A content-filtered response (choices == []) must not raise IndexError."""
    client_cls = _install_fake_openai(monkeypatch)
    monkeypatch.setattr(client_cls, "choices", [])
    adapter = _make_adapter()

    assert adapter.describe(b"\x89PNG x") == ""


def test_reasoning_deployment_uses_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gpt-5/o-series deployment must send reasoning_effort instead of temperature."""
    _install_fake_openai(monkeypatch)
    adapter = _make_adapter(deployment="gpt-5-vision-prod")

    adapter.describe(b"\x89PNG x")

    (call,) = adapter._ensure_client().calls
    assert "temperature" not in call
    assert call["reasoning_effort"] == "minimal"


def test_non_reasoning_deployment_uses_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gpt-4o-class deployment keeps the deterministic temperature=0.0."""
    _install_fake_openai(monkeypatch)
    adapter = _make_adapter(deployment="gpt-4o-vision")

    adapter.describe(b"\x89PNG x")

    (call,) = adapter._ensure_client().calls
    assert call["temperature"] == 0.0
    assert "reasoning_effort" not in call


def test_unsupported_temperature_400_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """An opaque non-reasoning name that is actually a reasoning model recovers via swap."""
    client_cls = _install_fake_openai(monkeypatch)
    monkeypatch.setattr(client_cls, "error", _FakeBadRequestError("temperature is not supported"))
    monkeypatch.setattr(client_cls, "error_once", True)
    adapter = _make_adapter(deployment="vision-prod")  # name does not look like a reasoning model

    out = adapter.describe(b"\x89PNG x")

    assert out == "a red square"
    calls = adapter._ensure_client().calls
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert calls[1].get("reasoning_effort") == "minimal"
    assert "temperature" not in calls[1]


def test_vendor_error_is_wrapped_as_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-parameter vendor failure is converted to StageError at the edge."""
    client_cls = _install_fake_openai(monkeypatch)
    monkeypatch.setattr(client_cls, "error", RuntimeError("rate limit"))
    adapter = _make_adapter()

    with pytest.raises(StageError):
        adapter.describe(b"\x89PNG x")
