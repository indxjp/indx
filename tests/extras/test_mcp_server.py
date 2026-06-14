"""Extras e2e for the **real MCP server**, exercised two ways against the official SDK.

The ``mcp`` package (the official Model Context Protocol SDK) is a dependency of
``fastmcp``, so installing ``indx[mcp]`` provides *both* ``fastmcp`` (the server kb.mcp()
builds) and ``mcp`` (the client used to drive it). Each test self-skips via
:func:`requires_lib` when ``mcp`` isn't importable, keeping the directory green on legs
where the extra wasn't installed.

* ``test_mcp_server_builds_tools`` — builds the in-process FastMCP server from a connected
  space and proves the three indx tools are wired, introspecting tool names defensively
  across FastMCP versions (with a connector-dispatch fallback).
* ``test_mcp_stdio_roundtrip`` — the real protocol e2e: launches ``indx mcp <archive>`` as
  a stdio subprocess and drives it through the official ``ClientSession``, proving the
  stdout stream is clean JSON-RPC (the banner goes to stderr).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from indx import DirectoryPipeline, __version__
from indx.agent import connect

pytestmark = pytest.mark.extras

ACME_DIR = Path(__file__).resolve().parents[1] / "corpus" / "acme_kb"
TOOL_NAMES = {"indx_search", "indx_overview", "indx_get_document"}


def _build_space():
    return DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl"
    ).run(ACME_DIR)


def _introspect_tool_names(server: object) -> set[str] | None:
    """Best-effort tool-name extraction across FastMCP versions; None if unsupported."""
    import anyio  # transitive dep of mcp; imported lazily so collection stays import-light

    # bundled mcp.server.fastmcp: async list_tools() returning objects with .name
    if hasattr(server, "list_tools"):
        try:
            tools = anyio.run(server.list_tools)
            return {t.name for t in tools}
        except Exception:
            pass
    # fastmcp v2: get_tools() may return a dict (name -> tool) or a list
    if hasattr(server, "get_tools"):
        try:
            tools = server.get_tools()  # type: ignore[attr-defined]
            if isinstance(tools, dict):
                return set(tools.keys())
            return {t.name for t in tools}
        except Exception:
            pass
    # fastmcp v2 internal manager
    mgr = getattr(server, "_tool_manager", None)
    if mgr is not None and hasattr(mgr, "list_tools"):
        try:
            tools = mgr.list_tools()
            return {t.name for t in tools}
        except Exception:
            pass
    return None


def test_mcp_server_builds_tools(requires_lib) -> None:
    requires_lib("mcp")
    space = _build_space()
    kb = connect(space, name="acme")

    server = kb.mcp()
    assert server is not None, "kb.mcp() must return a FastMCP server"

    names = _introspect_tool_names(server)
    if names is not None:
        assert names >= TOOL_NAMES, f"server is missing indx tools: got {names}"
    else:
        # No introspection path worked on this FastMCP version — still prove the wiring
        # through the single connector dispatch path the server delegates to.
        hits = kb.call("indx_search", {"query": "401k", "k": 1})
        assert hits["hits"], "indx_search dispatch returned no hits"


def test_mcp_stdio_roundtrip(requires_lib, tmp_path: Path) -> None:
    requires_lib("mcp")
    indx_bin = shutil.which("indx")
    if indx_bin is None:
        pytest.skip("the 'indx' console script is not on PATH")

    import anyio  # transitive dep of mcp; imported lazily so collection stays import-light
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    archive_path = tmp_path / "acme.indx"
    DirectoryPipeline(seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl").run(
        ACME_DIR, archive_path, name="acme"
    )
    assert archive_path.exists(), "build did not seal the .indx archive"

    params = StdioServerParameters(
        command=indx_bin,
        args=["mcp", str(archive_path), "--transport", "stdio"],
    )

    async def _roundtrip() -> None:
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    # serverInfo.version must advertise indx's version, not FastMCP's own
                    # package version (regression: bug #16).
                    assert init.serverInfo.version == __version__

                    tools = await session.list_tools()
                    names = {t.name for t in tools.tools}
                    assert names >= TOOL_NAMES, f"stdio server missing tools: {names}"

                    res = await session.call_tool(
                        "indx_search",
                        {"query": "retirement 401k matching contribution", "k": 1},
                    )
                    payload = json.loads(res.content[0].text)
                    assert payload["hits"][0]["source"].endswith("retirement-401k.md")

                    ov = await session.call_tool("indx_overview", {"sample": 2})
                    assert json.loads(ov.content[0].text)["documents"] == 24

                    gd = await session.call_tool(
                        "indx_get_document", {"path_or_id": "vector-store.md"}
                    )
                    assert json.loads(gd.content[0].text)["path"].endswith("vector-store.md")

    anyio.run(_roundtrip)
