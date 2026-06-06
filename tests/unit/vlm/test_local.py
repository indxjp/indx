"""Unit tests for :class:`indx.vlm.local.LocalVLM` (vendor ``httpx`` mocked, offline)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_json: Any = None,
    raise_on_post: Exception | None = None,
) -> dict[str, Any]:
    """Inject a fake ``httpx`` module and neutralise the dependency gate.

    Returns a dict capturing how the fake client was constructed and called so tests
    can assert on the request the adapter built.
    """
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._payload

    class _FakeClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def post(self, url: str, *, json: Any) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            if raise_on_post is not None:
                raise raise_on_post
            return _FakeResponse(response_json)

    fake = types.ModuleType("httpx")
    fake.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake)
    # Neutralise the dependency gate. The adapter binds ``require_extra`` by reference at
    # import time, so patch it where it is *used* (not just on indx.utils.lazy).
    import indx.vlm.local as local_mod

    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr(local_mod, "require_extra", lambda *a, **k: None)
    return captured


def _ok_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) env, constructing the adapter raises MissingExtraError."""
    from indx.vlm.local import LocalVLM

    with pytest.raises(MissingExtraError) as exc:
        LocalVLM()
    assert exc.value.extra == "vlm-local"


def test_satisfies_vlm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, response_json=_ok_response("x"))
    from indx.vlm.base import VLM
    from indx.vlm.local import LocalVLM

    assert isinstance(LocalVLM(), VLM)


def test_describe_returns_endpoint_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_httpx(monkeypatch, response_json=_ok_response("a red square"))
    from indx.vlm.local import LocalVLM

    result = LocalVLM(model="my-vlm").describe(b"\x89PNG-bytes", prompt="What is this?")

    assert result == "a red square"
    # The adapter POSTs to the OpenAI-compatible chat endpoint.
    assert captured["url"] == "/chat/completions"
    body = captured["json"]
    assert body["model"] == "my-vlm"
    assert body["temperature"] == 0.0  # determinism default
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_base_url_from_kwarg_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_httpx(monkeypatch, response_json=_ok_response("x"))
    from indx.vlm.local import LocalVLM

    vlm = LocalVLM(base_url="http://host:9999/v1/")
    vlm.describe(b"img")
    assert captured["base_url"] == "http://host:9999/v1"  # trailing slash stripped


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_httpx(monkeypatch, response_json=_ok_response("x"))
    monkeypatch.setenv("INDX_VLM_BASE_URL", "http://env-host:1234/v1")
    from indx.vlm.local import LocalVLM

    LocalVLM().describe(b"img")
    assert captured["base_url"] == "http://env-host:1234/v1"


def test_content_part_list_response_is_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": [{"text": "foo"}, {"text": "bar"}]}}]}
    _install_fake_httpx(monkeypatch, response_json=payload)
    from indx.vlm.local import LocalVLM

    assert LocalVLM().describe(b"img") == "foobar"


def test_default_prompt_used_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_httpx(monkeypatch, response_json=_ok_response("x"))
    from indx.vlm.local import LocalVLM

    LocalVLM().describe(b"img")
    assert captured["json"]["messages"][0]["content"][0]["text"]  # non-empty default


def test_transport_error_becomes_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, raise_on_post=RuntimeError("connection refused"))
    from indx.vlm.local import LocalVLM

    with pytest.raises(StageError):
        LocalVLM().describe(b"img")


def test_malformed_response_becomes_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, response_json={"unexpected": True})
    from indx.vlm.local import LocalVLM

    with pytest.raises(StageError):
        LocalVLM().describe(b"img")
