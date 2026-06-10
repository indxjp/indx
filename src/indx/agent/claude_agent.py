"""Claude Agent SDK adapter: expose a knowledge space as an in-process MCP server.

The Claude Agent SDK consumes tools as MCP servers. :func:`to_claude_mcp_server` builds an
*in-process* SDK MCP server (no subprocess, no socket) from the canonical operations; hand it
to ``ClaudeAgentOptions(mcp_servers={"indx": server})`` and the agent can search the space.

The ``claude_agent_sdk`` package is the optional ``claude-agent`` extra, imported lazily and
gated by :func:`~indx.utils.lazy.require_extra`; importing this module is always safe.
"""

from __future__ import annotations

from typing import Any

from indx.agent.connector import KnowledgeConnector
from indx.agent.schema import GET_DOCUMENT_TOOL, OVERVIEW_TOOL, SEARCH_TOOL
from indx.utils.lazy import require_extra


def _text_result(payload: Any) -> dict[str, Any]:
    """Wrap a JSON-able payload in the SDK's ``{"content": [{"type": "text", ...}]}`` shape."""
    import json

    return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}


def to_claude_mcp_server(connector: KnowledgeConnector, *, name: str = "indx") -> Any:
    """Build an in-process Claude Agent SDK MCP server exposing the space's tools."""
    require_extra("agent connector", "claude-agent", "claude-agent", "claude_agent_sdk")
    from claude_agent_sdk import (  # type: ignore[import-not-found]  # optional extra: claude-agent
        create_sdk_mcp_server,
        tool,
    )

    @tool(  # type: ignore[untyped-decorator]
        SEARCH_TOOL.name,
        SEARCH_TOOL.description,
        SEARCH_TOOL.parameters,
    )
    async def search(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            connector.call(
                SEARCH_TOOL.name,
                {
                    "query": args["query"],
                    "k": args.get("k", connector.default_k),
                    "doc_type": args.get("doc_type"),
                },
            )
        )

    @tool(OVERVIEW_TOOL.name, OVERVIEW_TOOL.description, OVERVIEW_TOOL.parameters)  # type: ignore[untyped-decorator]
    async def overview(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(connector.call(OVERVIEW_TOOL.name, {"sample": args.get("sample", 10)}))

    @tool(  # type: ignore[untyped-decorator]
        GET_DOCUMENT_TOOL.name,
        GET_DOCUMENT_TOOL.description,
        GET_DOCUMENT_TOOL.parameters,
    )
    async def get_document(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            connector.call(GET_DOCUMENT_TOOL.name, {"path_or_id": args["path_or_id"]})
        )

    return create_sdk_mcp_server(name=name, tools=[search, overview, get_document])
