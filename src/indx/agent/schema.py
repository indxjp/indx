"""Agent-facing I/O schemas and the canonical tool definitions.

These are pure Pydantic v2 models with **no vendor imports**, so importing this module is
always safe on a core-only install (file-architecture §5, coding-standards §6.3). Every
framework adapter (LangChain, OpenAI Agents SDK, Pydantic AI, Claude Agent SDK) and the MCP
server is built on top of the *same* three tool definitions below, so an agent sees one
consistent contract no matter how the knowledge space is plugged in.

The result models are deliberately **flat and JSON-primitive** — an LLM reads them straight
out of a tool result, and downstream stores/serializers never choke on nested vendor objects
(the same rule the output writers follow, coding-standards §6.2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- results


class Hit(BaseModel):
    """One retrieval result, flattened for an LLM to read directly.

    Mirrors the useful parts of a :class:`~indx.store.base.SearchHit` without the nested
    chunk/neighbor objects: the matched text, its similarity ``score``, and the provenance an
    agent needs to cite a source (path, folder, detected type, topics, tags).
    """

    chunk_id: str
    document_id: str
    score: float
    text: str
    source: str | None = None  # path relative to the indexed root (portable)
    folder: str = ""
    doc_type: str | None = None
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Adjacent chunk text, included only when context expansion is requested.
    context: list[str] = Field(default_factory=list)


class SearchResults(BaseModel):
    """The return shape of the ``indx_search`` tool."""

    query: str
    count: int
    hits: list[Hit] = Field(default_factory=list)


class DocumentCard(BaseModel):
    """A compact document summary — enough for an agent to decide if it's relevant."""

    id: str
    path: str
    doc_type: str | None = None
    folder: str = ""
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None


class SpaceOverview(BaseModel):
    """The return shape of the ``indx_overview`` tool: what this knowledge space contains."""

    name: str
    documents: int
    chunks: int
    relations: int
    embeddings: int
    embedding_model: str | None = None
    embedding_dim: int | None = None
    types: dict[str, int] = Field(default_factory=dict)
    sample_documents: list[DocumentCard] = Field(default_factory=list)


class DocumentDetail(DocumentCard):
    """The return shape of the ``indx_get_document`` tool: a card plus the full text."""

    chunk_count: int = 0
    text: str = ""


# --------------------------------------------------------------------------- tool defs


class ToolDef(BaseModel):
    """A framework-agnostic tool definition: name, description, and JSON-Schema parameters.

    Every adapter projects these into its own native shape — an OpenAI function spec, an
    Anthropic tool spec, a Pydantic AI / LangChain / OpenAI-Agents function, or an MCP tool —
    so the agent-facing contract is defined exactly once here.
    """

    name: str
    description: str
    parameters: dict[str, Any]


SEARCH_TOOL = ToolDef(
    name="indx_search",
    description=(
        "Semantic search over an indx knowledge space (a portable folder of documents "
        "turned AI-ready). Returns the most relevant text chunks with their source path, "
        "document type, and similarity score. Use this to ground answers in the indexed "
        "documents and to cite where a fact came from."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
                "default": 5,
                "minimum": 1,
            },
            "doc_type": {
                "type": ["string", "null"],
                "description": (
                    "Optional: restrict results to one detected document type "
                    "(e.g. 'policy', 'guide'). Call indx_overview to see available types."
                ),
                "default": None,
            },
        },
        "required": ["query"],
    },
)

OVERVIEW_TOOL = ToolDef(
    name="indx_overview",
    description=(
        "Describe the indx knowledge space: how many documents and chunks it holds, the "
        "embedding model, the histogram of document types, and a sample of documents with "
        "their summaries and topics. Call this first to learn what the space is about before "
        "searching it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sample": {
                "type": "integer",
                "description": "How many sample documents to include (default 10).",
                "default": 10,
                "minimum": 0,
            },
        },
        "required": [],
    },
)

GET_DOCUMENT_TOOL = ToolDef(
    name="indx_get_document",
    description=(
        "Fetch the full text and metadata of a single document by its path (e.g. "
        "'people/remote-work.md') or its document id. Use this after indx_search to read a "
        "promising source in full."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path_or_id": {
                "type": "string",
                "description": "The document's relative path or its document id.",
            },
        },
        "required": ["path_or_id"],
    },
)

#: The canonical tool set every adapter exposes, in a stable order.
TOOLS: list[ToolDef] = [SEARCH_TOOL, OVERVIEW_TOOL, GET_DOCUMENT_TOOL]
