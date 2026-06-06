"""Unit tests for the Claude Agent SDK adapter — ``claude_agent_sdk`` mocked, offline."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

import indx.agent.claude_agent  # noqa: F401  -- force-load before any require_extra patch
from indx.agent import KnowledgeConnector
from indx.errors import MissingExtraError


def _tool(name: str, description: str, schema: dict[str, Any]) -> Any:
    """Stand-in for ``claude_agent_sdk.tool``: tags the handler with its name/schema."""

    def decorator(fn: Any) -> Any:
        fn.tool_name = name
        fn.tool_schema = schema
        return fn

    return decorator


def _create_sdk_mcp_server(*, name: str, tools: list[Any]) -> Any:
    return types.SimpleNamespace(name=name, tools=tools)


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.agent.claude_agent.require_extra", lambda *a, **k: None)
    module = types.ModuleType("claude_agent_sdk")
    module.tool = _tool  # type: ignore[attr-defined]
    module.create_sdk_mcp_server = _create_sdk_mcp_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)


def test_claude_mcp_server_shape(fake_claude: None, kb: KnowledgeConnector) -> None:
    server = kb.claude()
    assert server.name == "handbook"
    assert [t.tool_name for t in server.tools] == [
        "indx_search",
        "indx_overview",
        "indx_get_document",
    ]

    # Handlers are async and return the SDK content-block shape wrapping our JSON payload.
    search = server.tools[0]
    out = asyncio.run(search({"query": "remote work", "k": 1}))
    assert out["content"][0]["type"] == "text"
    payload = json.loads(out["content"][0]["text"])
    assert payload["count"] >= 1
    assert payload["hits"][0]["source"] == "people/remote-work.md"


def test_claude_custom_name(fake_claude: None, kb: KnowledgeConnector) -> None:
    assert kb.claude(name="my-kb").name == "my-kb"


def test_claude_without_extra_raises(kb: KnowledgeConnector) -> None:
    with pytest.raises(MissingExtraError):
        kb.claude()
