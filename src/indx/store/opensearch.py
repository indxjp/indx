"""OpenSearchStore — Amazon OpenSearch Serverless vector store. Opt-in ``aws-opensearch``.

Wraps :class:`opensearchpy.OpenSearch` behind the tiny
:class:`~indx.store.base.VectorStore` Protocol (``name``, ``upsert``, ``search``,
``delete``). The heavy vendor SDK is imported lazily inside ``__init__`` so importing
this module is always safe; selecting the slot without the extra installed raises
:class:`~indx.errors.MissingExtraError` with an actionable
``pip install indx[aws-opensearch]`` hint (file-architecture §5, coding-standards §6).

Auth rides AWS SigV4 via ``AWSV4SignerAuth`` over credentials resolved by boto3's own
chain (env / profile / instance role) — keys are never read or logged here
(coding-standards §13). **Footgun encoded:** the SigV4 *service name* is ``"aoss"`` for
Serverless collections, not ``"es"``. Vendor types never leak past this module: ``Chunk``
is converted to/from an index ``_source`` document only at the edge (coding-standards §6.2),
and the kNN ``_score`` OpenSearch returns is already "higher is better", matching the
SearchHit contract with no inversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from opensearchpy import (  # type: ignore[import-not-found]  # optional extra: aws-opensearch
        OpenSearch,
    )

logger = logging.getLogger(__name__)

_INDEX = "indx"
_BATCH = 256
_SERVICE = "aoss"  # SigV4 service name for OpenSearch Serverless (NOT "es"; footgun §5)


class OpenSearchStore:
    """Vector store backed by Amazon OpenSearch Serverless (``aoss``).

    When ``dim`` is supplied (e.g. forwarded by the pipeline from the embedder's real
    width, or pinned via ``[store.opensearch] dim``) the index is created up-front with a
    ``knn_vector`` field of dimension ``dim`` and ``space_type="cosinesimil"``, so a
    subsequent :meth:`upsert`/:meth:`search` round-trip works immediately. When ``dim`` is
    ``None`` (an SDK user constructing the store directly without knowing the embedder's
    width) index creation is *deferred* until the first :meth:`upsert`, where it is sized
    to ``len(first_vector)`` — so the width always comes from a real vector, never a
    hard-coded guess (T10).

    Serverless vector collections disallow a custom ``_id`` on index, so the
    :class:`~indx.core.chunk.Chunk` id is carried inside the ``_source`` document
    (``chunk_id``) and used for upsert-by-id semantics and deletes; the remaining chunk
    fields ride along in the same document.

    Attributes:
        name: The registry key for this store (``"opensearch"``).
    """

    name = "opensearch"

    def __init__(
        self,
        *,
        host: str | None = None,
        region: str | None = None,
        index: str = _INDEX,
        dim: int | None = None,
    ) -> None:
        """Construct the store; create the backing index now or lazily.

        Args:
            host: The OpenSearch Serverless collection endpoint host (no scheme, no port),
                e.g. ``"abc123.us-east-1.aoss.amazonaws.com"``. Optional so the store can be
                resolved by the registry without args (the missing-extra gate fires first);
                when the extra is installed but ``host`` is absent, a :class:`StageError` is
                raised with an actionable message.
            region: AWS region for SigV4 signing (e.g. ``"us-east-1"``). When ``None``,
                boto3's own resolution (``AWS_REGION``/``AWS_DEFAULT_REGION``/profile)
                supplies it via the credentials chain.
            index: Index name to store chunks in; defaults to ``"indx"``.
            dim: Vector dimensionality the index is sized to. When ``None``, the index is
                created lazily on the first :meth:`upsert`, sized to the width of the first
                vector.

        Raises:
            MissingExtraError: If the ``aws-opensearch`` extra is not installed.
            StageError: If the client cannot be created or the index cannot be ensured.
        """
        # First statement: fail with one actionable message if the extra is absent,
        # so importing this module stays safe (coding-standards §6.3).
        require_extra("store", "opensearch", "aws-opensearch", "opensearchpy")
        if not host:
            raise StageError(
                "store",
                "OpenSearch requires a collection endpoint "
                "(set [store.opensearch] host= to the aoss host, e.g. "
                "'abc123.us-east-1.aoss.amazonaws.com').",
            )
        import boto3  # type: ignore[import-not-found]  # optional extra: aws-opensearch
        from opensearchpy import (  # optional extra: aws-opensearch (lazy; see TYPE_CHECKING)
            AWSV4SignerAuth,
            OpenSearch,
            RequestsHttpConnection,
        )

        self._index = index
        self._dim = dim
        try:
            # Credentials come from boto3's own chain (env / profile / role) — never read
            # or logged here (coding-standards §13). The signer service name is "aoss" for
            # Serverless, not "es" (footgun §5).
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, region, _SERVICE)
            self._client: OpenSearch = OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=auth,
                connection_class=RequestsHttpConnection,
                use_ssl=True,
                verify_certs=True,
            )
            # Pre-size the index only when a dim is known; otherwise defer to the first
            # upsert so it is sized to a real vector's width (T10 lazy sizing).
            if dim is not None and not self._client.indices.exists(index=self._index):
                self._ensure_index(dim)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"OpenSearch initialization failed: {exc}") from exc
        logger.debug("initialized OpenSearch store index=%s dim=%s", index, dim)

    def _ensure_index(self, dim: int) -> None:
        """Create the index with a ``knn_vector`` field sized to ``dim`` if it is absent.

        Idempotent and records the resolved ``dim`` so later upserts reuse it rather than
        re-checking width. Used both for up-front (``dim`` given) and lazy (first-upsert)
        sizing so there is a single sizing path.
        """
        if not self._client.indices.exists(index=self._index):
            self._client.indices.create(
                index=self._index,
                body={
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": dim,
                                "space_type": "cosinesimil",
                            },
                        }
                    },
                },
            )
        self._dim = dim

    def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert embedded chunks into the index, batched.

        Chunks without an ``embedding`` are skipped — a chunk may reach the store without a
        vector if the Embed stage was disabled, and indexing a ``None`` vector is an error.

        Serverless vector collections disallow a custom ``_id``, so the chunk id is carried
        inside ``_source`` (``chunk_id``) instead. To keep upsert-by-id semantics, any
        existing documents with the same ``chunk_id`` are deleted first.

        Args:
            chunks: The chunks to store. Each carries its vector and provenance.

        Raises:
            StageError: If the OpenSearch index call fails.
        """
        ready = [c for c in chunks if c.embedding is not None]
        if not ready:
            return
        try:
            # Deferred sizing: if the index was never sized at construction (dim was
            # unknown), size it now to the first real vector's width — not a guess (T10).
            if self._dim is None:
                self._ensure_index(len(ready[0].embedding or []))
            # Upsert-by-id: drop any prior docs carrying these chunk ids, then index.
            self.delete([c.id for c in ready])
            for i in range(0, len(ready), _BATCH):  # batch writes (coding-standards §14)
                for c in ready[i : i + _BATCH]:
                    self._client.index(
                        index=self._index,
                        body=_chunk_to_source(c),  # convert Chunk -> document AT THE EDGE
                    )
        except StageError:
            raise
        except Exception as exc:
            raise StageError("store", f"OpenSearch upsert failed: {exc}") from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending score order.

        Args:
            vector: The query embedding.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit`, highest cosine score first.
            Ties are broken on ``chunk.id`` so ordering is deterministic across runs.

        Raises:
            StageError: If the store was never sized (no ``dim`` and no prior upsert), or
                if the OpenSearch query call fails.
        """
        # Searching before the index exists (no dim given, nothing upserted yet) is a usage
        # error — surface a clean typed IndxError, not a raw vendor "index_not_found".
        if self._dim is None:
            raise StageError(
                "store",
                "OpenSearch store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        body = {"size": k, "query": {"knn": {"embedding": {"vector": vector, "k": k}}}}
        try:
            result = self._client.search(index=self._index, body=body)
        except Exception as exc:
            raise StageError("store", f"OpenSearch query failed: {exc}") from exc

        hits = [_hit_to_chunk_hit(h) for h in result["hits"]["hits"]]
        # Tie-break on chunk id so equal-score ties resolve identically (coding-standards §1.4).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete documents whose ``chunk_id`` matches, via a delete-by-query.

        Serverless collections carry the chunk id in ``_source`` rather than ``_id``, so
        removal is by a ``terms`` query on ``chunk_id`` rather than by document id.

        Args:
            chunk_ids: The ids of the chunks to remove. May be empty (a no-op).

        Raises:
            StageError: If the OpenSearch delete call fails.
        """
        if not chunk_ids:
            return
        try:
            self._client.delete_by_query(
                index=self._index,
                body={"query": {"terms": {"chunk_id": list(chunk_ids)}}},
            )
        except Exception as exc:
            raise StageError("store", f"OpenSearch delete failed: {exc}") from exc


def _chunk_to_source(chunk: Chunk) -> dict[str, Any]:
    """Convert a core :class:`Chunk` to an index ``_source`` document (vendor edge)."""
    return {
        "chunk_id": chunk.id,  # carried in _source: Serverless disallows custom _id (§5)
        "embedding": chunk.embedding,
        "doc_id": chunk.doc_id,
        "position": chunk.position,
        "text": chunk.text,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
    }


def _hit_to_chunk_hit(hit: dict[str, Any]) -> SearchHit:
    """Convert an OpenSearch search hit back to a core :class:`SearchHit` (vendor edge)."""
    source = hit.get("_source") or {}
    raw_vector = source.get("embedding")
    embedding = [float(v) for v in raw_vector] if raw_vector is not None else None
    chunk = Chunk(
        id=str(source["chunk_id"]),
        doc_id=source["doc_id"],
        position=source["position"],
        text=source["text"],
        prev_id=source.get("prev_id"),
        next_id=source.get("next_id"),
        embedding=embedding,
    )
    return SearchHit(chunk, float(hit["_score"]))
