"""Mocked unit tests for BedrockVLM — the boto3 SDK is absent, so it is faked.

The vendor ``boto3`` module is injected into ``sys.modules`` and the dependency gate is
neutralized so construction proceeds offline (coding-standards §11). The real dep-absent
construction is asserted in a separate test that does *not* neutralize the gate.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.vlm.base import VLM


class _FakeBedrockClient:
    reply: str | None = "a red square on a white background"

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        content: list[dict[str, Any]] = []
        if self.reply is not None:
            content.append({"text": self.reply})
        return {"output": {"message": {"content": content}}}


class _FakeSession:
    last_client: _FakeBedrockClient | None = None

    def __init__(self, *, profile_name: str | None = None) -> None:
        self.profile_name = profile_name

    def client(self, service: str, *, region_name: str | None = None) -> _FakeBedrockClient:
        client = _FakeBedrockClient()
        client.service = service  # type: ignore[attr-defined]
        client.region_name = region_name  # type: ignore[attr-defined]
        _FakeSession.last_client = client
        return client


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> type[_FakeBedrockClient]:
    """Inject a fake ``boto3`` module and neutralize the require_extra gate.

    The gate is neutralized on the adapter module itself. ``indx.vlm.bedrock`` binds
    ``require_extra`` by name at import time, so we import it first (before patching) to
    capture the real reference, then patch the bound name; monkeypatch restores it on
    teardown so the dep-absent test still sees the real gate.
    """
    import indx.vlm.bedrock as bedrock_mod

    _FakeBedrockClient.reply = "a red square on a white background"
    _FakeSession.last_client = None

    fake_module = types.ModuleType("boto3")
    fake_module.Session = _FakeSession  # type: ignore[attr-defined]
    fake_module.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)
    monkeypatch.setitem(sys.modules, "boto3", fake_module)
    monkeypatch.setattr(bedrock_mod, "require_extra", lambda *a, **k: None)
    return _FakeBedrockClient


def test_satisfies_vlm_protocol(fake_boto3: type[_FakeBedrockClient]) -> None:
    from indx.vlm.bedrock import BedrockVLM

    assert isinstance(BedrockVLM(), VLM)
    assert BedrockVLM().name == "bedrock"


def test_default_model_is_claude_inference_profile(
    fake_boto3: type[_FakeBedrockClient],
) -> None:
    from indx.vlm.bedrock import BedrockVLM

    assert BedrockVLM().model == "us.anthropic.claude-sonnet-4-6"
    assert BedrockVLM("us.amazon.nova-lite-v1:0").model == "us.amazon.nova-lite-v1:0"


def test_describe_returns_caption_text(fake_boto3: type[_FakeBedrockClient]) -> None:
    from indx.vlm.bedrock import BedrockVLM

    vlm = BedrockVLM()
    out = vlm.describe(b"\x89PNG fake bytes", prompt="What is this?")

    assert out == "a red square on a white background"


def test_describe_sends_raw_bytes_not_base64(
    fake_boto3: type[_FakeBedrockClient],
) -> None:
    from indx.vlm.bedrock import BedrockVLM

    vlm = BedrockVLM(region="us-east-1")
    image = b"\x89PNG fake bytes"
    vlm.describe(image, prompt="caption please")

    client = vlm._ensure_client()
    kwargs = client.last_kwargs
    assert kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"

    content = kwargs["messages"][0]["content"]
    text_part = next(p for p in content if "text" in p)
    image_part = next(p for p in content if "image" in p)
    assert text_part["text"] == "caption please"
    assert image_part["image"]["format"] == "png"
    # CRITICAL: raw bytes are passed verbatim — boto3 base64-encodes for us.
    assert image_part["image"]["source"]["bytes"] is image


def test_default_prompt_used_when_none(fake_boto3: type[_FakeBedrockClient]) -> None:
    from indx.vlm.bedrock import BedrockVLM

    vlm = BedrockVLM()
    vlm.describe(b"x")
    content = vlm._ensure_client().last_kwargs["messages"][0]["content"]
    text_part = next(p for p in content if "text" in p)
    assert text_part["text"] == "Describe this image in detail."


def test_describe_handles_empty_content(
    monkeypatch: pytest.MonkeyPatch, fake_boto3: type[_FakeBedrockClient]
) -> None:
    from indx.vlm.bedrock import BedrockVLM

    monkeypatch.setattr(_FakeBedrockClient, "reply", None)
    assert BedrockVLM().describe(b"x") == ""


def test_invalid_image_format_raises_stage_error(
    fake_boto3: type[_FakeBedrockClient],
) -> None:
    from indx.vlm.bedrock import BedrockVLM

    with pytest.raises(StageError) as exc_info:
        BedrockVLM(image_format="tiff")
    assert exc_info.value.stage == "enrich"


def test_describe_wraps_vendor_failure_in_stage_error(
    monkeypatch: pytest.MonkeyPatch, fake_boto3: type[_FakeBedrockClient]
) -> None:
    from indx.vlm.bedrock import BedrockVLM

    def _boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("ValidationException: use an inference profile")

    monkeypatch.setattr(_FakeBedrockClient, "converse", _boom)
    with pytest.raises(StageError) as exc_info:
        BedrockVLM().describe(b"x")
    assert exc_info.value.stage == "enrich"


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.vlm.bedrock import BedrockVLM

    # Guard against fake-module leakage from other tests: the real env has no boto3.
    assert "boto3" not in sys.modules
    assert importlib.util.find_spec("boto3") is None

    with pytest.raises(MissingExtraError) as exc_info:
        BedrockVLM()
    assert exc_info.value.extra == "aws"
