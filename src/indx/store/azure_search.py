"""AzureAISearchStore — the default Azure vector store. Requires the ``azure`` extra.

Wraps the Azure AI Search clients (:class:`azure.search.documents.SearchClient` and
:class:`azure.search.documents.indexes.SearchIndexClient`) behind the tiny
:class:`~indx.store.base.VectorStore` Protocol (``name``, ``upsert``, ``search``,
``delete``). The heavy ``azure-search-documents`` SDK is imported lazily inside
``__init__`` so importing this module is always safe; selecting the slot without the extra
installed raises :class:`~indx.errors.MissingExtraError` with an actionable
``pip install indx[azure]`` hint (file-architecture §5, coding-standards §6).

Vendor types never leak past this module: :class:`~indx.core.chunk.Chunk` is converted
to/from a search document (a plain dict) only at the edge (coding-standards §6.2), and the
``@search.score`` Azure returns is already "higher is better", matching the SearchHit
contract with no inversion. Credentials are taken from the environment
(``AZURE_SEARCH_API_KEY``) and handed to the SDK's own ``AzureKeyCredential`` — never read
or logged directly (coding-standards §13).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from azure.search.documents import (  # type: ignore[import-not-found]  # optional extra: azure
        SearchClient,
    )

logger = logging.getLogger(__name__)

_SLOT = "store"
_NAME = "azure-search"
_EXTRA = "azure"

_VECTOR_FIELD = "contentVector"
_PROFILE = "vp"
_ALGORITHM = "alg"
_SELECT = ["id", "content", "doc_id", "position", "prev_id", "next_id"]
# Azure AI Search caps indexing requests at 1000 documents per call; batch to stay under it.
_BATCH = 1000


class AzureAISearchStore:
    """Vector store backed by Azure AI Search (the default Azure store).

    When ``dim`` is supplied (e.g. forwarded by the pipeline from the embedder's real width,
    or pinned via ``[store.azure-search] dim``) the search index is created up-front sized to
    ``dim`` so a subsequent :meth:`upsert`/:meth:`search` round-trip works immediately. When
    ``dim`` is ``None`` (an SDK user constructing the store directly without knowing the
    embedder's width) index creation is *deferred* until the first :meth:`upsert`, where it
    is sized to ``len(first_vector)`` — so the width always comes from a real vector, never a
    hard-coded guess (T10). Document keys are the :class:`~indx.core.chunk.Chunk` ids
    (strings pass straight through); the remaining chunk fields ride along as document fields.

    Credentials come from the environment (``AZURE_SEARCH_API_KEY``) and are handed to the
    SDK's ``AzureKeyCredential``; the admin key is required to create the index and upload,
    a query key suffices for search.

    Attributes:
        name: The registry key for this store (``"azure-search"``).
    """

    name = "azure-search"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        index_name: str | None = None,
        dim: int | None = None,
    ) -> None:
        """Construct the store; the backing index is created now or lazily.

        Args:
            endpoint: Search service endpoint URL. Defaults to the
                ``AZURE_SEARCH_SERVICE_ENDPOINT`` environment variable.
            key: Admin/query API key. Defaults to the ``AZURE_SEARCH_API_KEY`` environment
                variable. Handed to the SDK's ``AzureKeyCredential``; never logged.
            index_name: Name of the search index. Defaults to the
                ``AZURE_SEARCH_INDEX_NAME`` environment variable.
            dim: Vector dimensionality the index is sized to. When ``None``, the index is
                created lazily on the first :meth:`upsert`, sized to the width of the first
                vector.

        Raises:
            MissingExtraError: If the ``azure`` extra is not installed.
            StageError: If the endpoint, key, or index name is missing.
        """
        # First statement: fail with one actionable message if the extra is absent, so
        # importing this module stays safe (coding-standards §6.3).
        require_extra(_SLOT, _NAME, _EXTRA, "azure.search.documents")

        self._endpoint = endpoint or os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
        self._key = key or os.environ.get("AZURE_SEARCH_API_KEY")
        self._index_name = index_name or os.environ.get("AZURE_SEARCH_INDEX_NAME")
        self._dim = dim
        self._created = False

        if not self._endpoint:
            raise StageError(
                "store",
                "Azure AI Search requires a service endpoint "
                "(set AZURE_SEARCH_SERVICE_ENDPOINT or pass endpoint=).",
            )
        if not self._key:
            raise StageError(
                "store",
                "Azure AI Search requires an API key (set AZURE_SEARCH_API_KEY or pass key=).",
            )
        if not self._index_name:
            raise StageError(
                "store",
                "Azure AI Search requires an index name "
                "(set AZURE_SEARCH_INDEX_NAME or pass index_name=).",
            )
        logger.debug(
            "initialized Azure AI Search store endpoint=%s index=%s dim=%s",
            self._endpoint,
            self._index_name,
            dim,
        )

    def _search_client(self) -> SearchClient:
        """Build a per-call ``SearchClient`` for the configured index (vendor edge)."""
        # optional extra: azure (lazy; see TYPE_CHECKING import)
        from azure.core.credentials import (  # type: ignore[import-not-found]  # optional extra: azure
            AzureKeyCredential,
        )
        from azure.search.documents import SearchClient

        return SearchClient(
            endpoint=self._endpoint,
            index_name=self._index_name,
            credential=AzureKeyCredential(self._key),
        )

    def _ensure_index(self, dim: int) -> None:
        """Create (or update) the search index sized to ``dim`` if not already done.

        Idempotent (``create_or_update_index``) and records the resolved ``dim`` so later
        upserts reuse it rather than re-checking width. Used both for up-front (``dim``
        given) and lazy (first-upsert) sizing so there is a single sizing path.

        Raises:
            StageError: If the index cannot be created.
        """
        if self._created:
            return
        # optional extra: azure (lazy; see TYPE_CHECKING import)
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import (  # type: ignore[import-not-found]  # optional extra: azure
            SearchIndexClient,
        )
        from azure.search.documents.indexes.models import (  # type: ignore[import-not-found]  # optional extra: azure
            HnswAlgorithmConfiguration,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="doc_id", type=SearchFieldDataType.String),
            SearchField(name="position", type=SearchFieldDataType.Int32),
            SearchField(name="prev_id", type=SearchFieldDataType.String),
            SearchField(name="next_id", type=SearchFieldDataType.String),
            SearchField(
                name=_VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dim,
                vector_search_profile_name=_PROFILE,
            ),
        ]
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name=_PROFILE, algorithm_configuration_name=_ALGORITHM)],
            algorithms=[HnswAlgorithmConfiguration(name=_ALGORITHM)],
        )
        index = SearchIndex(name=self._index_name, fields=fields, vector_search=vector_search)
        try:
            client = SearchIndexClient(
                endpoint=self._endpoint, credential=AzureKeyCredential(self._key)
            )
            client.create_or_update_index(index)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Azure AI Search index setup failed: {exc}") from exc
        self._dim = dim
        self._created = True

    def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert embedded chunks into the index.

        Chunks without an ``embedding`` are skipped — a chunk may reach the store without a
        vector if the Embed stage was disabled, and pushing ``None`` into a vector field is
        an error.

        Args:
            chunks: The chunks to store. Each carries its vector and provenance.

        Raises:
            StageError: If the Azure upload call fails.
        """
        ready = [c for c in chunks if c.embedding is not None]
        documents = [_chunk_to_document(c) for c in ready]  # convert Chunk -> doc AT THE EDGE
        if not documents:
            return
        # Deferred sizing: if the index was never sized at construction (dim was unknown),
        # size it now to the first real vector's width — not a guess (T10).
        if self._dim is None:
            self._ensure_index(len(ready[0].embedding or []))
        else:
            self._ensure_index(self._dim)
        try:
            client = self._search_client()
            for start in range(0, len(documents), _BATCH):
                client.merge_or_upload_documents(documents[start : start + _BATCH])
        except Exception as exc:
            raise StageError("store", f"Azure AI Search upsert failed: {exc}") from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending score order.

        Args:
            vector: The query embedding.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit`, highest ``@search.score`` first.
            Ties are broken on ``chunk.id`` so ordering is deterministic across runs.

        Raises:
            StageError: If the store was never sized (no ``dim`` and no prior upsert), or if
                the Azure query call fails.
        """
        # Searching before the index exists (no dim given, nothing upserted yet) is a usage
        # error — surface a clean typed IndxError, not a raw vendor "index not found".
        if self._dim is None:
            raise StageError(
                "store",
                "Azure AI Search store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        # optional extra: azure (lazy; see TYPE_CHECKING import)
        from azure.search.documents.models import (  # type: ignore[import-not-found]  # optional extra: azure
            VectorizedQuery,
        )

        try:
            results = self._search_client().search(
                search_text=None,
                vector_queries=[
                    VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields=_VECTOR_FIELD)
                ],
                select=_SELECT,
            )
            rows = list(results)
        except Exception as exc:
            raise StageError("store", f"Azure AI Search query failed: {exc}") from exc

        hits = [_row_to_chunk_hit(r) for r in rows]
        # Tie-break on chunk id so equal-score ties resolve identically (coding-standards §1.4).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents by chunk id.

        Args:
            chunk_ids: The ids of the chunks (documents) to remove.

        Raises:
            StageError: If the Azure delete call fails.
        """
        if not chunk_ids:
            return
        try:
            client = self._search_client()
            for start in range(0, len(chunk_ids), _BATCH):
                client.delete_documents([{"id": i} for i in chunk_ids[start : start + _BATCH]])
        except Exception as exc:
            raise StageError("store", f"Azure AI Search delete failed: {exc}") from exc

    def close(self) -> None:
        """Release any resources the store holds.

        Unlike the retained-client stores, this store keeps **no** persistent vendor
        client on ``self``: a fresh ``SearchClient``/``SearchIndexClient`` is built and
        discarded inside each call (:meth:`_search_client`, :meth:`_ensure_index`), so
        there is no long-lived connection pool to drop here. This override is therefore an
        explicit, idempotent no-op that completes the lifecycle contract callers rely on
        (``with AzureAISearchStore(...) as store``); it is safe to call more than once.
        """
        return None

    def __enter__(self) -> AzureAISearchStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _chunk_to_document(chunk: Chunk) -> dict[str, Any]:
    """Convert a core :class:`Chunk` to an Azure search document (vendor edge)."""
    return {
        "id": chunk.id,
        "content": chunk.text,
        _VECTOR_FIELD: chunk.embedding,
        "doc_id": chunk.doc_id,
        "position": chunk.position,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
    }


def _row_to_chunk_hit(row: Any) -> SearchHit:
    """Convert an Azure search result row (dict-like) back to a core SearchHit (vendor edge)."""
    chunk = Chunk(
        id=str(row["id"]),
        doc_id=row["doc_id"],
        position=row["position"],
        text=row["content"],
        prev_id=row.get("prev_id"),
        next_id=row.get("next_id"),
    )
    return SearchHit(chunk, float(row["@search.score"]))
