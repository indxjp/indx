"""Mocked unit tests for :class:`indx.llm.bedrock.BedrockLLM`.

The ``boto3`` SDK is not installed in this environment, so it is faked: a stand-in ``boto3``
module is injected into ``sys.modules`` and the dependency gate is neutralized. Tests are
deterministic and offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.llm.base import LLM
from indx.llm.bedrock import BedrockLLM


class _FakeBedrockClient:
    """Minimal stand-in for a boto3 ``bedrock-runtime`` client that records call kwargs."""

    reply: str | None = "fake-bedrock-response"
    raise_exc: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        content: list[dict[str, Any]] = []
        if self.reply is not None:
            content = [{"text": self.reply}]
        return {"output": {"message": {"content": content}}}


class _FakeBoto3:
    """Records the requested service + region and returns a single shared client stub."""

    def __init__(self) -> None:
        self.client_calls: list[dict[str, Any]] = []
        self.last_client: _FakeBedrockClient | None = None

    def client(self, service: str, **kwargs: Any) -> _FakeBedrockClient:
        self.client_calls.append({"service": service, **kwargs})
        self.last_client = _FakeBedrockClient(**kwargs)
        return self.last_client


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch) -> _FakeBoto3:
    """Neutralize the extra gate and inject a fake ``boto3`` module."""
    # bedrock.py binds require_extra at import time (``from ... import require_extra``),
    # so patch the name in both the source module and the consuming module.
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.llm.bedrock.require_extra", lambda *a, **k: None)
    fake = _FakeBoto3()
    fake_module = types.ModuleType("boto3")
    fake_module.client = fake.client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)
    return fake


def test_satisfies_llm_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    adapter = BedrockLLM(region="us-east-1")
    assert adapter.name == "bedrock"
    assert isinstance(adapter, LLM)


def test_default_model_is_us_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    adapter = BedrockLLM()
    assert adapter.model == "us.anthropic.claude-sonnet-4-6"


def test_complete_round_trips_through_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_boto3(monkeypatch)
    adapter = BedrockLLM(model="us.anthropic.claude-sonnet-4-6", region="eu-west-1")

    out = adapter.complete("hello", system="be brief", max_tokens=64, temperature=0.0)

    assert out == "fake-bedrock-response"
    # Client constructed for bedrock-runtime in the configured region.
    (client_call,) = fake.client_calls
    assert client_call["service"] == "bedrock-runtime"
    assert client_call["region_name"] == "eu-west-1"
    # The request used the model id and Bedrock-shaped system/messages/inferenceConfig.
    assert fake.last_client is not None
    (call,) = fake.last_client.calls
    assert call["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert call["system"] == [{"text": "be brief"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert call["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.0}


def test_complete_without_system_passes_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_boto3(monkeypatch)
    adapter = BedrockLLM(region="us-east-1")
    adapter.complete("just user")
    assert fake.last_client is not None
    (call,) = fake.last_client.calls
    assert call["system"] == []


def test_picks_first_text_content_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    adapter = BedrockLLM(region="us-east-1")
    client = adapter._ensure_client()

    def converse(**kwargs: Any) -> dict[str, Any]:
        # A tool-use block precedes the text block; the adapter must skip it defensively.
        return {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "x"}},
                        {"text": "the answer"},
                    ]
                }
            }
        }

    monkeypatch.setattr(client, "converse", converse)
    assert adapter.complete("x") == "the answer"


def test_no_text_block_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    monkeypatch.setattr(_FakeBedrockClient, "reply", None)
    adapter = BedrockLLM(region="us-east-1")
    assert adapter.complete("x") == ""


def test_reads_region_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_boto3(monkeypatch)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    adapter = BedrockLLM()
    adapter.complete("hi")
    (client_call,) = fake.client_calls
    assert client_call["region_name"] == "ap-southeast-2"


def test_vendor_failure_wrapped_in_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_boto3(monkeypatch)
    monkeypatch.setattr(_FakeBedrockClient, "raise_exc", RuntimeError("boom"))
    adapter = BedrockLLM(region="us-east-1")
    with pytest.raises(StageError) as excinfo:
        adapter.complete("x")
    assert excinfo.value.stage == "enrich"


def test_inference_profile_validation_gives_actionable_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_boto3(monkeypatch)
    exc = RuntimeError(
        "ValidationException: Invocation of model ID ... isn't supported. "
        "Retry your request with the ID or ARN of an inference profile that contains this model."
    )
    monkeypatch.setattr(_FakeBedrockClient, "raise_exc", exc)
    adapter = BedrockLLM(model="anthropic.claude-3-5-sonnet-20240620-v1:0", region="us-east-1")
    with pytest.raises(StageError) as excinfo:
        adapter.complete("x")
    assert excinfo.value.stage == "enrich"
    assert "us." in str(excinfo.value)


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real (dep-absent) environment, construction must fail with MissingExtraError.

    require_extra is NOT neutralized here, so the absent ``boto3`` / ``aws`` extra surfaces.
    """
    with pytest.raises(MissingExtraError):
        BedrockLLM(region="us-east-1")
