"""OpenAI Agents SDK adapter: expose a knowledge space as ``function_tool``s.

Wraps each canonical operation with :func:`agents.function_tool`, whose schema is derived
from the wrapped function's signature, type hints, and docstring. Drop the returned tools
into an ``agents.Agent(tools=...)``.

The ``agents`` package is the optional ``openai-agents`` extra, imported lazily and gated by
:func:`~indx.utils.lazy.require_extra`; importing this module is always safe.
"""

from __future__ import annotations

from typing import Any

from indx.agent.connector import KnowledgeConnector
from indx.agent.schema import GET_DOCUMENT_TOOL, OVERVIEW_TOOL, SEARCH_TOOL
from indx.utils.lazy import require_extra


def to_openai_agent_tools(connector: KnowledgeConnector) -> list[Any]:
    """Build OpenAI Agents SDK ``function_tool``s (search / overview / get_document)."""
    require_extra("agent connector", "openai-agents", "openai-agents", "agents")
    from agents import (  # type: ignore[import-not-found]  # optional extra: openai-agents
        function_tool,
    )

    def indx_search(query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        """Semantic search over the indx knowledge space; returns ranked, cited text chunks."""
        return connector.call(SEARCH_TOOL.name, {"query": query, "k": k, "doc_type": doc_type})

    def indx_overview(sample: int = 10) -> dict[str, Any]:
        """Describe the knowledge space: counts, document types, and sample documents."""
        return connector.call(OVERVIEW_TOOL.name, {"sample": sample})

    def indx_get_document(path_or_id: str) -> dict[str, Any]:
        """Fetch one document's full text and metadata by path or id."""
        return connector.call(GET_DOCUMENT_TOOL.name, {"path_or_id": path_or_id})

    return [
        function_tool(indx_search),
        function_tool(indx_overview),
        function_tool(indx_get_document),
    ]
