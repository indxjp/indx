"""KnowledgeConnector — plug a knowledge space into any AI agent, USB-drive style.

A ``.indx`` archive is a portable knowledge space: the "USB drive" you carry between
machines and agents. :class:`KnowledgeConnector` is the plug. It wraps a
:class:`~indx.core.knowledge_space.KnowledgeSpace` and exposes a tiny, stable set of
agent operations — **search**, **overview**, **get_document** — plus one-call adapters that
hand those operations to whichever agent framework you use:

* :meth:`~KnowledgeConnector.langchain` / :meth:`~KnowledgeConnector.langchain_retriever`
* :meth:`~KnowledgeConnector.openai`        (OpenAI Agents SDK)
* :meth:`~KnowledgeConnector.pydantic_ai`   (Pydantic AI)
* :meth:`~KnowledgeConnector.claude`        (Claude Agent SDK, in-process MCP server)
* :meth:`~KnowledgeConnector.mcp`           (Model Context Protocol — Mastra & any client)

For frameworks not covered, :meth:`openai_schema` / :meth:`anthropic_schema` emit raw
tool specs and :meth:`call` dispatches a tool call by name — enough to wire the bare
Chat Completions / Messages API by hand.

This module imports **no vendor SDKs at top level**; every adapter is imported lazily inside
its method and gated by :func:`~indx.utils.lazy.require_extra`, so ``import indx.agent`` is
safe on a bare ``pip install indx`` (file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from indx.agent.schema import (
    GET_DOCUMENT_TOOL,
    OVERVIEW_TOOL,
    SEARCH_TOOL,
    TOOLS,
    DocumentCard,
    DocumentDetail,
    Hit,
    SearchResults,
    SpaceOverview,
    ToolDef,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indx.core.knowledge_space import KnowledgeSpace


class KnowledgeConnector:
    """An agent-ready handle on a single knowledge space.

    Construct it directly from an in-memory space, or use :meth:`open` / the module-level
    :func:`connect` to load a ``.indx`` archive (or an output directory) from disk.

    Attributes:
        name: A short identifier for the space, surfaced to the agent in tool descriptions
            and used as the default MCP server name.
        default_k: The number of hits :meth:`search` returns when ``k`` is unset.
        with_context: When true, every :meth:`search` hit carries its neighbor chunks' text
            in ``hit.context`` for wider grounding windows.
    """

    def __init__(
        self,
        space: KnowledgeSpace,
        *,
        name: str = "indx",
        default_k: int = 5,
        with_context: bool = False,
    ) -> None:
        self._space = space
        self.name = name
        self.default_k = default_k
        self.with_context = with_context

    @classmethod
    def open(
        cls,
        source: str | Path | KnowledgeSpace,
        *,
        name: str | None = None,
        default_k: int = 5,
        with_context: bool = False,
    ) -> KnowledgeConnector:
        """Load a knowledge space from ``source`` and wrap it.

        ``source`` may be an already-loaded :class:`KnowledgeSpace`, a path to a ``.indx``
        archive, or an output directory containing one (the same inputs ``indx inspect`` /
        ``indx query`` accept). ``name`` defaults to the archive's file stem.
        """
        from indx.core.knowledge_space import KnowledgeSpace

        if isinstance(source, KnowledgeSpace):
            space = source
            label = name or "indx"
        else:
            # Reuse the CLI loader so the connector accepts every on-disk shape the CLI does
            # (a .indx file, a directory holding one, or a jsonl output directory).
            from indx.cli._render import load_space

            path = Path(source)
            space = load_space(path)
            label = name or (path.stem if path.is_file() else path.name or "indx")

        return cls(space, name=label, default_k=default_k, with_context=with_context)

    @property
    def space(self) -> KnowledgeSpace:
        """The wrapped :class:`KnowledgeSpace` (read access for advanced callers)."""
        return self._space

    # ------------------------------------------------------------------ operations

    def search(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
        *,
        with_context: bool | None = None,
    ) -> SearchResults:
        """Semantic search over the space; the backbone of the ``indx_search`` tool.

        Routes through :meth:`KnowledgeSpace.search` (CLI ⇄ SDK parity), then flattens each hit
        into a JSON-primitive :class:`~indx.agent.schema.Hit`. When ``doc_type`` is given,
        results are filtered to that detected type, over-fetching first so a full ``k`` can
        still come back.
        """
        try:
            k = self.default_k if k is None else max(int(k), 1)
        except (TypeError, ValueError):
            k = self.default_k
        want_context = self.with_context if with_context is None else with_context

        raw = self._space.search(query, k=k * 5 if doc_type else k)

        hits: list[Hit] = []
        for hit in raw:
            doc = self._space.document(hit.chunk.doc_id)
            hit_type = (hit.source.type if hit.source else None) or (doc.doc_type if doc else None)
            if doc_type and (hit_type or "unknown") != doc_type:
                continue
            hits.append(
                Hit(
                    chunk_id=hit.chunk.id,
                    document_id=hit.chunk.doc_id,
                    score=hit.score,
                    text=hit.chunk.text,
                    source=(hit.source.path if hit.source else (doc.path if doc else None)),
                    folder=(hit.source.folder if hit.source else (doc.folder if doc else "")),
                    doc_type=hit_type,
                    topics=list(doc.topics) if doc else [],
                    tags=list(doc.tags) if doc else [],
                    context=[c.text for c in hit.neighbors] if want_context else [],
                )
            )
            if len(hits) >= k:
                break

        return SearchResults(query=query, count=len(hits), hits=hits)

    def overview(self, sample: int = 10) -> SpaceOverview:
        """Summarize the space; the backbone of the ``indx_overview`` tool."""
        stats = self._space.stats
        try:
            n = max(int(sample), 0)
        except (TypeError, ValueError):
            n = 10
        cards = [self._card(doc) for doc in self._space.documents()[:n]]
        return SpaceOverview(
            name=self.name,
            documents=stats.documents,
            chunks=stats.chunks,
            relations=stats.relations,
            embeddings=stats.embeddings,
            embedding_model=self._space.manifest.embedding_model,
            embedding_dim=stats.embed_dim,
            types=dict(stats.types),
            sample_documents=cards,
        )

    def get_document(self, path_or_id: str) -> DocumentDetail | None:
        """Fetch one document's full text + metadata; the backbone of ``indx_get_document``.

        Resolves ``path_or_id`` against document ids first, then exact paths, then a path
        suffix match (so ``remote-work.md`` finds ``people/remote-work.md``). Returns ``None``
        when nothing matches.
        """
        doc = self._space.document(path_or_id)
        if doc is None:
            docs = self._space.documents()
            doc = next((d for d in docs if d.path == path_or_id), None)
            if doc is None:
                doc = next(
                    (d for d in docs if d.path == path_or_id or d.path.endswith("/" + path_or_id)),
                    None,
                )
        if doc is None:
            return None

        chunks = sorted(self._space.chunks_for(doc.id), key=lambda c: c.position)
        card = self._card(doc)
        return DocumentDetail(
            **card.model_dump(),
            chunk_count=len(chunks),
            text="\n\n".join(c.text for c in chunks),
        )

    @staticmethod
    def _card(doc: Any) -> DocumentCard:
        return DocumentCard(
            id=doc.id,
            path=doc.path,
            doc_type=doc.doc_type,
            folder=doc.folder,
            topics=list(doc.topics),
            tags=list(doc.tags),
            summary=doc.summary,
        )

    # ------------------------------------------------------------------ raw specs

    def tools(self) -> list[ToolDef]:
        """The canonical, framework-agnostic tool definitions for this space."""
        return list(TOOLS)

    def openai_schema(self) -> list[dict[str, Any]]:
        """Tool specs in OpenAI Chat Completions / Responses ``tools=[...]`` shape."""
        return [{"type": "function", "function": t.model_dump()} for t in self.tools()]

    def anthropic_schema(self) -> list[dict[str, Any]]:
        """Tool specs in Anthropic Messages API ``tools=[...]`` shape."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self.tools()
        ]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a tool call by ``name`` and return a JSON-able result.

        This is the single execution path every adapter and the MCP server funnel through, so
        a tool behaves identically regardless of which framework invoked it. Unknown names
        raise :class:`ValueError`.
        """
        args = arguments or {}
        if name == SEARCH_TOOL.name:
            query = args.get("query")
            if not isinstance(query, str) or not query:
                return {"error": "missing required argument 'query'"}
            return self.search(
                query=query,
                k=args.get("k"),
                doc_type=args.get("doc_type"),
            ).model_dump(mode="json")
        if name == OVERVIEW_TOOL.name:
            sample = args.get("sample")
            sample = 10 if sample is None else sample
            return self.overview(sample=sample).model_dump(mode="json")
        if name == GET_DOCUMENT_TOOL.name:
            path_or_id = args.get("path_or_id")
            if not isinstance(path_or_id, str) or not path_or_id:
                return {"error": "missing required argument 'path_or_id'"}
            detail = self.get_document(path_or_id)
            if detail is None:
                return {"error": f"no document matching {path_or_id!r}"}
            return detail.model_dump(mode="json")
        raise ValueError(f"unknown tool {name!r}; known tools: {[t.name for t in TOOLS]}")

    # ------------------------------------------------------------------ adapters

    def langchain(self) -> list[Any]:
        """Return LangChain ``StructuredTool``s for this space (needs ``indx[langchain]``)."""
        from indx.agent.langchain import to_langchain_tools

        return to_langchain_tools(self)

    def langchain_retriever(self, k: int | None = None) -> Any:
        """Return a LangChain ``BaseRetriever`` over this space (needs ``indx[langchain]``)."""
        from indx.agent.langchain import to_langchain_retriever

        return to_langchain_retriever(self, k=k or self.default_k)

    def openai(self) -> list[Any]:
        """Return OpenAI Agents SDK ``function_tool``s (needs ``indx[openai-agents]``)."""
        from indx.agent.openai_agents import to_openai_agent_tools

        return to_openai_agent_tools(self)

    def pydantic_ai(self) -> list[Any]:
        """Return Pydantic AI ``Tool``s for this space (needs ``indx[pydantic-ai]``)."""
        from indx.agent.pydantic_ai import to_pydantic_ai_tools

        return to_pydantic_ai_tools(self)

    def claude(self, *, name: str | None = None) -> Any:
        """Return an in-process Claude Agent SDK MCP server (needs ``indx[claude-agent]``)."""
        from indx.agent.claude_agent import to_claude_mcp_server

        return to_claude_mcp_server(self, name=name or self.name)

    def mcp(self, *, name: str | None = None) -> Any:
        """Return a ``FastMCP`` server exposing this space (needs ``indx[mcp]``)."""
        from indx.agent.mcp import build_mcp_server

        return build_mcp_server(self, name=name or self.name)

    def serve(self, *, transport: str = "stdio", name: str | None = None) -> None:
        """Run an MCP server over ``transport`` until interrupted (needs ``indx[mcp]``).

        This is what ``indx mcp <archive>`` calls: it turns the knowledge space into a live
        MCP endpoint that Claude Desktop, Mastra, Cursor, or any MCP client can connect to.
        """
        self.mcp(name=name).run(transport=transport)


def connect(
    source: str | Path | KnowledgeSpace,
    *,
    name: str | None = None,
    default_k: int = 5,
    with_context: bool = False,
) -> KnowledgeConnector:
    """Plug a knowledge space into an agent in one line — ``connect("space.indx")``.

    A thin alias for :meth:`KnowledgeConnector.open`; the headline entry point of
    :mod:`indx.agent`.
    """
    return KnowledgeConnector.open(
        source, name=name, default_k=default_k, with_context=with_context
    )
