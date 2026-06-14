"""Unit tests for the MCP server adapter — ``mcp`` package mocked, offline."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.agent import KnowledgeConnector
from indx.agent import mcp as mcp_module
from indx.errors import MissingExtraError


class _FastMCP:
    """Stand-in for ``mcp.server.fastmcp.FastMCP``: registers tools and records ``run``."""

    def __init__(self, name: str, version: str | None = None) -> None:
        self.name = name
        self.version = version
        self.tools: dict[str, Any] = {}
        self.ran_transport: str | None = None

    def add_tool(self, fn: Any, *, name: str | None = None) -> Any:
        self.tools[name or fn.__name__] = fn
        return fn

    def run(self, transport: str = "stdio") -> None:
        self.ran_transport = transport


@pytest.fixture
def fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the standalone ``fastmcp`` package — the preferred FastMCP source."""
    module = types.ModuleType("fastmcp")
    module.FastMCP = _FastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastmcp", module)


@pytest.fixture
def fake_bundled_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject only the bundled ``mcp.server.fastmcp`` to exercise the fallback path."""
    pkg = types.ModuleType("mcp")
    server_pkg = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = _FastMCP  # type: ignore[attr-defined]
    for name, module in {
        "mcp": pkg,
        "mcp.server": server_pkg,
        "mcp.server.fastmcp": fastmcp,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_mcp_server_registers_three_tools(fake_fastmcp: None, kb: KnowledgeConnector) -> None:
    server = kb.mcp()
    assert server.name == "handbook"
    assert set(server.tools) == {"indx_search", "indx_overview", "indx_get_document"}

    # Registered tools dispatch through the connector and return JSON-able dicts.
    assert server.tools["indx_search"](query="remote", k=1)["count"] >= 1
    assert server.tools["indx_overview"]()["documents"] == 2
    assert server.tools["indx_get_document"](path_or_id="d1")["chunk_count"] == 2
    assert server.tools["indx_get_document"](path_or_id="ghost") == {"error": "not found"}


def test_prefers_standalone_fastmcp(fake_fastmcp: None) -> None:
    # With the standalone package present, that is the FastMCP we load.
    from fastmcp import FastMCP

    assert mcp_module._load_fastmcp() is FastMCP


def test_falls_back_to_bundled_fastmcp(fake_bundled_mcp: None) -> None:
    # No standalone `fastmcp`; the bundled mcp.server.fastmcp.FastMCP is used instead.
    from mcp.server.fastmcp import FastMCP

    assert mcp_module._load_fastmcp() is FastMCP


def test_server_run_records_transport(fake_fastmcp: None, kb: KnowledgeConnector) -> None:
    server = mcp_module.build_mcp_server(kb)
    server.run(transport="sse")
    assert server.ran_transport == "sse"


def test_connector_serve_does_not_raise(fake_fastmcp: None, kb: KnowledgeConnector) -> None:
    # connector.serve() builds a server and calls run(); the fake run() is a no-op.
    kb.serve(transport="stdio")


def test_mcp_without_extra_raises(kb: KnowledgeConnector) -> None:
    # Neither `fastmcp` nor `mcp` is installed in the test env → clean MissingExtraError.
    with pytest.raises(MissingExtraError):
        kb.mcp()
