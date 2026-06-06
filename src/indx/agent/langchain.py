"""LangChain adapter: expose a knowledge space as tools and a retriever.

Two integration points, both built on :class:`~indx.agent.connector.KnowledgeConnector`:

* :func:`to_langchain_tools` — ``StructuredTool``s an agent can call (search / overview /
  get_document), the agentic path.
* :func:`to_langchain_retriever` — a ``BaseRetriever`` that returns LangChain ``Document``s,
  the classic RAG path (drop it into any retrieval chain).

``langchain-core`` is the optional ``langchain`` extra, imported lazily and gated by
:func:`~indx.utils.lazy.require_extra`; importing this module is always safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from indx.agent.connector import KnowledgeConnector
from indx.agent.schema import GET_DOCUMENT_TOOL, OVERVIEW_TOOL, SEARCH_TOOL
from indx.utils.lazy import require_extra

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.callbacks import (  # type: ignore[import-not-found]  # optional extra: langchain
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import (  # type: ignore[import-not-found]  # optional extra: langchain
        Document,
    )
    from langchain_core.tools import (  # type: ignore[import-not-found]  # optional extra: langchain
        StructuredTool,
    )


def to_langchain_tools(connector: KnowledgeConnector) -> list[StructuredTool]:
    """Build LangChain ``StructuredTool``s (search / overview / get_document) for the space."""
    require_extra("agent connector", "langchain", "langchain", "langchain_core")
    from langchain_core.tools import StructuredTool  # optional extra: langchain

    def _search(query: str, k: int = 5, doc_type: str | None = None) -> dict[str, Any]:
        return connector.call(SEARCH_TOOL.name, {"query": query, "k": k, "doc_type": doc_type})

    def _overview(sample: int = 10) -> dict[str, Any]:
        return connector.call(OVERVIEW_TOOL.name, {"sample": sample})

    def _get_document(path_or_id: str) -> dict[str, Any]:
        return connector.call(GET_DOCUMENT_TOOL.name, {"path_or_id": path_or_id})

    return [
        StructuredTool.from_function(
            func=_search, name=SEARCH_TOOL.name, description=SEARCH_TOOL.description
        ),
        StructuredTool.from_function(
            func=_overview, name=OVERVIEW_TOOL.name, description=OVERVIEW_TOOL.description
        ),
        StructuredTool.from_function(
            func=_get_document,
            name=GET_DOCUMENT_TOOL.name,
            description=GET_DOCUMENT_TOOL.description,
        ),
    ]


def to_langchain_retriever(connector: KnowledgeConnector, *, k: int = 5) -> Any:
    """Build a LangChain ``BaseRetriever`` that returns ``Document``s from the space.

    Each retrieved ``Document`` carries the chunk text as ``page_content`` and the hit's
    provenance (source path, document type, score, topics, tags) as JSON-primitive
    ``metadata`` — the same metadata shape the ``langchain`` output writer emits.
    """
    require_extra("agent connector", "langchain", "langchain", "langchain_core")
    from langchain_core.documents import Document  # optional extra: langchain
    from langchain_core.retrievers import (  # type: ignore[import-not-found]  # optional extra: langchain
        BaseRetriever,
    )

    class IndxRetriever(BaseRetriever):  # type: ignore[misc]  # BaseRetriever is Any without the extra
        """Retrieve indx knowledge-space chunks as LangChain ``Document``s."""

        connector: Any
        k: int = 5

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> list[Document]:
            results = self.connector.search(query, k=self.k)
            return [
                Document(
                    id=hit.chunk_id,
                    page_content=hit.text,
                    metadata={
                        "doc_id": hit.document_id,
                        "score": hit.score,
                        "source": hit.source,
                        "folder": hit.folder,
                        "doc_type": hit.doc_type,
                        "topics": hit.topics,
                        "tags": hit.tags,
                    },
                )
                for hit in results.hits
            ]

    return IndxRetriever(connector=connector, k=k)
