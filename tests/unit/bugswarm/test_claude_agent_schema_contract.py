"""Regression: the Claude adapter must hand the SDK the canonical JSON-Schema.

Bug: ``claude_agent.py`` passed bare Python-type dicts (``{"query": str, ...}``) as the
SDK tool ``input_schema``. The real ``claude_agent_sdk`` then marks *every* property
``required`` and treats ``str`` as a non-nullable ``{"type": "string"}`` — so ``indx_search``
advertised ``k`` and ``doc_type`` as required (and ``doc_type`` as non-nullable) and
``indx_overview`` advertised ``sample`` as required, diverging from the single canonical
contract in ``schema.py`` that every other adapter honours.

The fix passes ``SEARCH_TOOL.parameters`` / ``OVERVIEW_TOOL.parameters`` /
``GET_DOCUMENT_TOOL.parameters`` verbatim. This test captures the schema each tool is
registered with and asserts it equals the canonical one (correct ``required`` list,
nullable ``doc_type``, defaults preserved). Fully offline: ``claude_agent_sdk`` is faked.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.agent.claude_agent  # noqa: F401  -- force-load before any require_extra patch
from indx.agent.connector import KnowledgeConnector
from indx.agent.schema import GET_DOCUMENT_TOOL, OVERVIEW_TOOL, SEARCH_TOOL
from indx.core.knowledge_space import KnowledgeSpace, Manifest


def _tool(name: str, description: str, schema: dict[str, Any]) -> Any:
    """Stand-in for ``claude_agent_sdk.tool``: record the schema it was registered with."""

    def decorator(fn: Any) -> Any:
        fn.tool_name = name
        fn.tool_schema = schema
        return fn

    return decorator


def _create_sdk_mcp_server(*, name: str, tools: list[Any]) -> Any:
    return types.SimpleNamespace(name=name, tools=tools)


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.agent.claude_agent.require_extra", lambda *a, **k: None)
    module = types.ModuleType("claude_agent_sdk")
    module.tool = _tool  # type: ignore[attr-defined]
    module.create_sdk_mcp_server = _create_sdk_mcp_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)


@pytest.fixture
def connector() -> KnowledgeConnector:
    space = KnowledgeSpace(manifest=Manifest(components={"embedder": "hash"}))
    return KnowledgeConnector(space, name="handbook")


def test_tools_registered_with_canonical_schema(
    fake_claude: None, connector: KnowledgeConnector
) -> None:
    server = connector.claude()
    schemas = {t.tool_name: t.tool_schema for t in server.tools}

    # Each tool gets the exact canonical JSON-Schema object — not a bare Python-type dict.
    assert schemas["indx_search"] == SEARCH_TOOL.parameters
    assert schemas["indx_overview"] == OVERVIEW_TOOL.parameters
    assert schemas["indx_get_document"] == GET_DOCUMENT_TOOL.parameters


def test_optional_params_not_required_and_doc_type_nullable(
    fake_claude: None, connector: KnowledgeConnector
) -> None:
    server = connector.claude()
    schemas = {t.tool_name: t.tool_schema for t in server.tools}

    search = schemas["indx_search"]
    # Only ``query`` is required; ``k`` and ``doc_type`` stay optional.
    assert search["required"] == ["query"]
    assert "k" not in search["required"]
    assert "doc_type" not in search["required"]
    # ``doc_type`` must remain nullable, not a bare non-null string.
    assert search["properties"]["doc_type"]["type"] == ["string", "null"]
    assert search["properties"]["k"]["default"] == 5

    # ``sample`` is optional on overview, never forced.
    assert schemas["indx_overview"]["required"] == []
