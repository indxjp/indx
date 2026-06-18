"""QdrantStore — the default vector store. Requires the ``qdrant`` extra.

Wraps :class:`qdrant_client.QdrantClient` behind the tiny
:class:`~indx.store.base.VectorStore` Protocol (``name``, ``upsert``, ``search``,
``delete``). The heavy vendor SDK is imported lazily inside ``__init__`` so importing
this module is always safe; selecting the slot without the extra installed raises
:class:`~indx.errors.MissingExtraError` with an actionable ``pip install indx[qdrant]``
hint (file-architecture §5, coding-standards §6).

The same client runs embedded on a local path (no server, air-gapped) and against a
remote server when a ``url`` is given — local-first by default, scalable when needed.
Vendor types never leak past this module: ``Chunk`` is converted to/from a Qdrant point
payload only at the edge (coding-standards §6.2), and the cosine ``score`` Qdrant returns
is already "higher is better", matching the SearchHit contract with no inversion.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from qdrant_client import (  # type: ignore[import-not-found]  # optional extra: qdrant
        QdrantClient,
    )

logger = logging.getLogger(__name__)

_COLLECTION = "indx"
_BATCH = 256
_DEFAULT_PATH = "./space.qdrant"
_CHUNK_ID_KEY = "chunk_id"


def _point_id(chunk_id: str) -> str:
    """Map an arbitrary chunk id to a valid Qdrant point id (a UUID string).

    Qdrant accepts only an unsigned integer or a UUID string as a point id; chunk ids
    are 16-char truncated SHA-256 hex strings (see :func:`indx.utils.hashing.stable_hash`)
    which are neither. Deriving a deterministic UUIDv5 keeps the mapping stable across runs
    so re-upsert and delete address the same point; the original id rides in the payload.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


class QdrantStore:
    """Vector store backed by Qdrant (embedded on-disk or a remote server).

    When ``dim`` is supplied (e.g. forwarded by the pipeline from the embedder's real
    width, or pinned via ``[store.qdrant] dim``) the collection is created up-front sized
    to ``dim`` with cosine distance, so a subsequent :meth:`upsert`/:meth:`search`
    round-trip works immediately. When ``dim`` is ``None`` (an SDK user constructing the
    store directly without knowing the embedder's width) collection creation is *deferred*
    until the first :meth:`upsert`, where it is sized to ``len(first_vector)`` — so the
    width always comes from a real vector, never a hard-coded guess (T10). Qdrant point ids
    must be a UUID or unsigned integer, so each :class:`~indx.core.chunk.Chunk` id is mapped
    to a deterministic UUID (:func:`_point_id`) and the original id rides in the payload; the
    remaining chunk fields ride along too.

    The collection name is configurable (``[store.qdrant] collection`` / ``collection=``,
    defaulting to ``"indx"``) so each space can own an isolated collection rather than every
    corpus piling into one global name. When the named collection already exists at a
    *different* width than the embedder produces, it is recreated at the new width instead of
    leaving a wrong-width collection that every upsert dies on with a cryptic vendor 400 —
    so swapping the embedder model (or running a local preset on a server that previously ran
    a cloud build) no longer breaks at the store stage after doing all the upstream work.

    Attributes:
        name: The registry key for this store (``"qdrant"``).
    """

    name = "qdrant"

    def __init__(
        self,
        *,
        path: str | None = None,
        url: str | None = None,
        dim: int | None = None,
        collection: str | None = None,
    ) -> None:
        """Construct the store; create the backing collection now or lazily.

        Args:
            path: On-disk directory for the embedded (no-server) client. Used when
                ``url`` is not given; defaults to ``"./space.qdrant"``.
            url: Remote Qdrant server URL (e.g. ``"http://localhost:6333"``). When set,
                the client connects to the server instead of opening an embedded path.
            dim: Vector dimensionality the collection is sized to. When ``None``, the
                collection is created lazily on the first :meth:`upsert`, sized to the
                width of the first vector.
            collection: Name of the backing collection (``[store.qdrant] collection``).
                Defaults to ``"indx"``. Give each space its own name to keep corpora
                isolated and let a width change be recreated in place rather than colliding
                with another corpus' collection.

        Raises:
            MissingExtraError: If the ``qdrant`` extra is not installed.
            StageError: If the client cannot be created or the collection cannot be
                ensured.
        """
        # First statement: fail with one actionable message if the extra is absent,
        # so importing this module stays safe (coding-standards §6.3).
        require_extra("store", "qdrant", "qdrant", "qdrant_client")
        from qdrant_client import (  # optional extra: qdrant (lazy; see TYPE_CHECKING import)
            QdrantClient,
            models,
        )

        self._models = models
        self._dim = dim
        self._collection = collection or _COLLECTION
        try:
            # Local-first default: on-disk, no server. ``url`` overrides to a server.
            self._client: QdrantClient = (
                QdrantClient(url=url) if url else QdrantClient(path=path or _DEFAULT_PATH)
            )
            # Pre-size the collection only when a dim is known; otherwise defer to the
            # first upsert so it is sized to a real vector's width (T10 lazy sizing).
            # When the dim is known, ``_ensure_collection`` also recreates a pre-existing
            # collection whose width differs — so a re-build with a different embedder no
            # longer dies on a wrong-width upsert (issue #24).
            # On reopen of an already-populated on-disk collection (dim unknown), recover
            # the persisted width so search() before any upsert no longer raises spuriously.
            if dim is not None:
                self._ensure_collection(dim)
            elif self._client.collection_exists(self._collection):
                info = self._client.get_collection(self._collection)
                self._dim = info.config.params.vectors.size
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Qdrant initialization failed: {exc}") from exc
        logger.debug(
            "initialized Qdrant store collection=%s dim=%s remote=%s",
            self._collection,
            dim,
            url is not None,
        )

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection sized to ``dim``, recreating it if its width differs.

        Idempotent and records the resolved ``dim`` so later upserts reuse it rather than
        re-checking width. Used both for up-front (``dim`` given) and lazy (first-upsert)
        sizing so there is a single sizing path. When the collection already exists at a
        different width, it is dropped and recreated at ``dim`` — safe now that the name is
        space-scoped — rather than left wrong-width for an upsert to die on (issue #24).
        """
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            if info.config.params.vectors.size == dim:
                self._dim = dim
                return
            # Width mismatch: the collection was sized to a different embedder. Drop and
            # recreate at the new width (delete+create is the non-deprecated recreate).
            logger.info(
                "recreating Qdrant collection %s: width %s -> %s",
                self._collection,
                info.config.params.vectors.size,
                dim,
            )
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config=self._models.VectorParams(
                size=dim, distance=self._models.Distance.COSINE
            ),
        )
        self._dim = dim

    def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert embedded chunks into the collection, batched.

        Chunks without an ``embedding`` are skipped — a chunk may reach the store
        without a vector if the Embed stage was disabled, and pushing ``None`` into a
        vector field is an error.

        Args:
            chunks: The chunks to store. Each carries its vector and provenance.

        Raises:
            StageError: If the Qdrant upsert call fails.
        """
        m = self._models
        ready = [c for c in chunks if c.embedding is not None]
        points = [
            m.PointStruct(
                # Qdrant point ids must be a UUID or unsigned int, so map the chunk id to a
                # deterministic UUID; the original id is carried in the payload.
                id=_point_id(c.id),
                vector=c.embedding,
                payload=_chunk_to_payload(c),  # convert Chunk -> payload AT THE EDGE
            )
            for c in ready
        ]
        if not points:
            return
        try:
            # Deferred sizing: if the collection was never sized at construction (dim was
            # unknown), size it now to the first real vector's width — not a guess (T10).
            if self._dim is None:
                self._ensure_collection(len(ready[0].embedding or []))
            for i in range(0, len(points), _BATCH):  # batch upserts (coding-standards §14)
                # wait=True makes the write durable before returning, so a query in the
                # same run sees the data (determinism, coding-standards §1.4).
                self._client.upsert(self._collection, points=points[i : i + _BATCH], wait=True)
        except Exception as exc:
            raise StageError("store", f"Qdrant upsert failed: {exc}") from exc

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
                if the Qdrant query call fails.
        """
        # Searching before the collection exists (no dim given, nothing upserted yet) is a
        # usage error — surface a clean typed IndxError, not a raw vendor "not found".
        if self._dim is None:
            raise StageError(
                "store",
                "Qdrant store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        try:
            result = self._client.query_points(
                self._collection, query=vector, limit=k, with_payload=True, with_vectors=True
            )
        except Exception as exc:
            raise StageError("store", f"Qdrant query failed: {exc}") from exc

        hits = [_point_to_chunk_hit(p) for p in result.points]
        # Tie-break on chunk id so equal-score ties resolve identically (coding-standards §1.4).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete points by chunk id.

        Args:
            chunk_ids: The ids of the chunks (points) to remove.

        Raises:
            StageError: If the Qdrant delete call fails.
        """
        if not chunk_ids:
            return
        try:
            self._client.delete(
                self._collection,
                points_selector=self._models.PointIdsList(
                    points=[_point_id(cid) for cid in chunk_ids]
                ),
            )
        except Exception as exc:
            raise StageError("store", f"Qdrant delete failed: {exc}") from exc

    def close(self) -> None:
        """Release the backing :class:`qdrant_client.QdrantClient`.

        The embedded (on-disk) client holds a local storage handle and the remote client a
        pooled HTTP/gRPC connection; a store that is never closed leaks them until GC.
        Best-effort: call the vendor's own ``close()`` if present, then drop the reference.
        Idempotent — safe to call more than once. After ``close()`` the store must not be
        used again.
        """
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                closer = getattr(client, "close", None)
                if callable(closer):
                    closer()
            except Exception:  # best-effort teardown; never raise from close()
                logger.debug("Qdrant client close failed", exc_info=True)
            self._client = None


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Convert a core :class:`Chunk` to a Qdrant point payload (vendor edge)."""
    return {
        # The point id is a derived UUID, so keep the real chunk id in the payload to read
        # back at search time (the chunk-id contract must survive the round-trip).
        _CHUNK_ID_KEY: chunk.id,
        "doc_id": chunk.doc_id,
        "position": chunk.position,
        "text": chunk.text,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
    }


def _point_to_chunk_hit(point: Any) -> SearchHit:
    """Convert a Qdrant scored point back to a core :class:`SearchHit` (vendor edge)."""
    payload = point.payload or {}
    raw_vector = getattr(point, "vector", None)
    embedding = [float(v) for v in raw_vector] if raw_vector is not None else None
    chunk = Chunk(
        # Recover the original chunk id from the payload; the point id is a derived UUID.
        id=str(payload.get(_CHUNK_ID_KEY, point.id)),
        doc_id=payload["doc_id"],
        position=payload["position"],
        text=payload["text"],
        prev_id=payload.get("prev_id"),
        next_id=payload.get("next_id"),
        embedding=embedding,
    )
    return SearchHit(chunk, float(point.score))
