"""MCP server: serve a knowledge space over the Model Context Protocol.

MCP is the universal connector — Claude Desktop, Cursor, Mastra (TypeScript), and any other
MCP client speak it, so one ``indx mcp <archive>`` command plugs a knowledge space into all
of them, no Python glue on the client side. :func:`build_mcp_server` builds a ``FastMCP``
server exposing the canonical search / overview / get_document tools;
:meth:`KnowledgeConnector.serve` runs it.

FastMCP does the heavy lifting: it derives each tool's JSON schema from the handler's typed
signature and owns the transport loop. We prefer the standalone, batteries-included
``fastmcp`` package (v2) and fall back to the ``FastMCP`` bundled in the official ``mcp`` SDK
(v1) — either satisfies the ``indx[mcp]`` extra. Imports are lazy, so importing this module is
always safe on a core-only install.
"""

from __future__ import annotations

from typing import Any

from indx._version import __version__
from indx.agent.connector import KnowledgeConnector
from indx.errors import MissingExtraError


def _load_fastmcp() -> Any:
    """Return a ``FastMCP`` class, preferring the standalone ``fastmcp`` over the bundled one.

    Raises :class:`MissingExtraError` (the standard ``pip install indx[mcp]`` message) if
    neither is installed — mirroring :func:`~indx.utils.lazy.require_extra` for the
    "either of two modules" case it can't express directly.
    """
    try:
        from fastmcp import FastMCP  # type: ignore[import-not-found]  # optional extra: mcp

        return FastMCP
    except ModuleNotFoundError:
        pass
    try:
        from mcp.server.fastmcp import (  # type: ignore[import-not-found]  # optional extra: mcp
            FastMCP,
        )

        return FastMCP
    except ModuleNotFoundError:
        raise MissingExtraError(slot="agent connector", name="mcp", extra="mcp") from None


def build_mcp_server(connector: KnowledgeConnector, *, name: str | None = None) -> Any:
    """Build a ``FastMCP`` server exposing the space's tools (search/overview/get_document).

    Tools are registered with ``add_tool`` (rather than the ``@server.tool`` decorator) so the
    handler functions keep their static types — mypy stays strict over this module.
    """
    fast_mcp = _load_fastmcp()
    # Pass indx's own version so the MCP initialize handshake advertises serverInfo.version
    # as indx's own ``__version__``, not FastMCP's package version — both FastMCP variants
    # accept it.
    server = fast_mcp(name or connector.name, version=__version__)

    def indx_search(query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        """Semantic search over the indx knowledge space."""
        return connector.search(query, k=k, doc_type=doc_type).model_dump(mode="json")

    def indx_overview(sample: int = 10) -> dict[str, Any]:
        """Describe the knowledge space: counts, types, sample documents."""
        return connector.overview(sample=sample).model_dump(mode="json")

    def indx_get_document(path_or_id: str) -> dict[str, Any]:
        """Fetch one document's full text and metadata by path or id."""
        detail = connector.get_document(path_or_id)
        return detail.model_dump(mode="json") if detail else {"error": "not found"}

    server.add_tool(indx_search)
    server.add_tool(indx_overview)
    server.add_tool(indx_get_document)
    return server
