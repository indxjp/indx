"""Regression test: Bedrock Cohere branch raises StageError on a dict response that
lacks the ``float`` key, instead of a bare KeyError that escapes the typed-error
boundary.

The ``boto3`` SDK is intentionally absent in this environment, so it is mocked: a fake
``boto3`` module is injected into ``sys.modules`` and the dependency gate
(:func:`indx.utils.lazy.require_extra`) is neutralized so construction proceeds.

All tests are deterministic and offline.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from indx.embed.bedrock import BedrockEmbedder
from indx.errors import StageError


class _FakeBody:
    """Mimics the streaming ``["body"]`` object: ``.read()`` yields JSON bytes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _NoFloatClient:
    """Bedrock client whose Cohere response is a dict without the ``float`` key."""

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:  # noqa: N803
        # A non-float embedding-type dict (e.g. int8) — malformed for this adapter.
        return {"body": _FakeBody({"embeddings": {"int8": [[1, 2, 3]]}})}


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    """Neutralize the extra gate and inject a fake ``boto3`` returning ``client``."""
    monkeypatch.setattr("indx.embed.bedrock.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("boto3")
    fake_module.client = lambda *a, **k: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)


def test_cohere_dict_without_float_raises_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cohere dict response lacking ``float`` raises StageError, not KeyError."""
    _install_fake_boto3(monkeypatch, _NoFloatClient())
    embedder = BedrockEmbedder(model="cohere.embed-english-v3")

    with pytest.raises(StageError) as excinfo:
        embedder.embed(["one"])

    assert "float" in str(excinfo.value)
