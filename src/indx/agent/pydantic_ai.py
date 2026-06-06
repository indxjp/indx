"""Pydantic AI adapter: expose a knowledge space as ``pydantic_ai.Tool``s.

Wraps each canonical operation in a :class:`pydantic_ai.Tool`; pass the list to
``Agent(tools=...)`` or register them with ``agent.tool``. Pydantic AI builds the JSON
schema from the wrapped function's typed signature and docstring.

The ``pydantic_ai`` package is the optional ``pydantic-ai`` extra, imported lazily and gated
by :func:`~indx.utils.lazy.require_extra`; importing this module is always safe.
"""

from __future__ import annotations

from typing import Any

from indx.agent.connector import KnowledgeConnector
from indx.agent.schema import GET_DOCUMENT_TOOL, OVERVIEW_TOOL, SEARCH_TOOL
from indx.utils.lazy import require_extra


def to_pydantic_ai_tools(connector: KnowledgeConnector) -> list[Any]:
    """Build Pydantic AI ``Tool``s (search / overview / get_document) for the space."""
    require_extra("agent connector", "pydantic-ai", "pydantic-ai", "pydantic_ai")
    from pydantic_ai import (  # type: ignore[import-not-found]  # optional extra: pydantic-ai
        Tool,
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
        Tool(indx_search, name=SEARCH_TOOL.name, description=SEARCH_TOOL.description),
        Tool(indx_overview, name=OVERVIEW_TOOL.name, description=OVERVIEW_TOOL.description),
        Tool(
            indx_get_document,
            name=GET_DOCUMENT_TOOL.name,
            description=GET_DOCUMENT_TOOL.description,
        ),
    ]
