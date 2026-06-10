"""ChromaStore — Chroma vector store. Requires the ``chroma`` extra.

Wraps a persistent :mod:`chromadb` collection behind the tiny
:class:`~indx.store.base.VectorStore` Protocol (``upsert`` / ``search`` /
``delete``). The heavy vendor SDK is imported lazily inside ``__init__`` so
importing this module is always safe; selecting the slot without the extra
installed raises :class:`~indx.errors.MissingExtraError` with an actionable
``pip install indx[chroma]`` hint (file-architecture §5, coding-standards §6).

Vendor types never leak past this module: ``Chunk`` is converted to Chroma's
parallel ``ids``/``embeddings``/``documents``/``metadatas`` lists on the way in,
and Chroma query results are converted back to core :class:`~indx.core.chunk.Chunk`
and :class:`~indx.store.base.SearchHit` objects on the way out (coding-standards §6.2).
Chroma reports cosine *distance*, so scores are inverted to the "higher is better,
descending" SearchHit contract, with a deterministic tie-break on ``chunk.id``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    import chromadb  # type: ignore[import-not-found]  # optional extra: chroma

logger = logging.getLogger(__name__)

# Single fixed collection per space; the .indx archive owns its own Chroma path.
_COLLECTION = "indx"
# Upsert in slices so a large chunk list is one round-trip per batch, not per chunk.
_BATCH = 256


class ChromaStore:
    """Vector store backed by an embedded, on-disk Chroma collection.

    The collection is created with cosine space (Chroma defaults to L2) so scores
    align with the other stores. Explicit vectors are always supplied, so the
    collection's own embedding function is disabled.

    Attributes:
        name: The registry key for this store (``"chroma"``).
    """

    name = "chroma"

    def __init__(self, path: str = "./chroma") -> None:
        """Construct the store and open (or create) its Chroma collection.

        Args:
            path: On-disk directory for the embedded Chroma client. Defaults to
                ``"./chroma"``.

        Raises:
            MissingExtraError: If the ``chroma`` extra is not installed.
            StageError: If the Chroma client or collection cannot be opened.
        """
        # First statement: fail with one actionable message if the extra is absent,
        # so importing this module stays safe (coding-standards §6.3).
        require_extra("store", "chroma", "chroma", "chromadb")
        import chromadb  # optional extra: chroma (lazy; see TYPE_CHECKING import above)

        logger.debug("initializing Chroma store path=%s collection=%s", path, _COLLECTION)
        try:
            self._client: chromadb.api.ClientAPI = chromadb.PersistentClient(path=path)
            self._col = self._client.get_or_create_collection(
                _COLLECTION,
                # We pass explicit vectors; never let Chroma try to embed text itself.
                embedding_function=None,
                # Default space is L2 — set cosine so scores match the other stores.
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Chroma collection could not be opened: {exc}") from exc

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update embedded chunks in the collection.

        Chunks without an embedding are skipped (Embed may have been disabled);
        never push ``None`` into a backend that expects float vectors. Writes are
        batched to avoid one round-trip per chunk.

        Args:
            chunks: The chunks to persist. Only those with a non-``None``
                ``embedding`` are written.

        Raises:
            StageError: If a Chroma upsert call fails.
        """
        ready = [c for c in chunks if c.embedding is not None]
        if not ready:
            return
        try:
            for start in range(0, len(ready), _BATCH):
                batch = ready[start : start + _BATCH]
                self._col.upsert(
                    ids=[c.id for c in batch],
                    embeddings=[c.embedding for c in batch],
                    documents=[c.text for c in batch],
                    # Chroma metadata values must be primitives and cannot be None:
                    # store "" and convert back to None on read.
                    metadatas=[
                        {
                            "doc_id": c.doc_id,
                            "position": c.position,
                            "prev_id": c.prev_id or "",
                            "next_id": c.next_id or "",
                        }
                        for c in batch
                    ],
                )
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Chroma upsert failed: {exc}") from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending score order.

        Chroma returns cosine *distance* (0 = identical); it is inverted to a
        similarity so the result honors the cross-store "higher is better" contract.
        Ties are broken on ``chunk.id`` for deterministic ordering (NFR-DET-1).

        Args:
            vector: The query embedding.
            k: The maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit` objects, highest score
            first.

        Raises:
            StageError: If the Chroma query fails.
        """
        try:
            result = self._col.query(
                query_embeddings=[vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Chroma query failed: {exc}") from exc

        # Chroma nests results per query embedding — always index [0].
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits = [
            SearchHit(_result_to_chunk(cid, doc, md), 1.0 - float(dist))
            for cid, doc, md, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks from the collection by id.

        Args:
            chunk_ids: The ids of the chunks to delete. Unknown ids are ignored.

        Raises:
            StageError: If the Chroma delete call fails.
        """
        if not chunk_ids:
            return
        try:
            self._col.delete(ids=chunk_ids)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Chroma delete failed: {exc}") from exc

    def close(self) -> None:
        """Release the underlying Chroma client.

        Chroma's persistent client holds on-disk SQLite/segment resources; not
        releasing it leaks handles and can lock the path against a second client.
        Idempotent: safe to call more than once. After ``close()`` the store must
        not be used again.
        """
        client = getattr(self, "_client", None)
        if client is not None:
            # Prefer the vendor's own teardown if the installed version exposes one;
            # otherwise drop the reference so the client is finalized promptly by GC.
            closer = getattr(client, "_system", None)
            try:
                if closer is not None and hasattr(closer, "stop"):
                    closer.stop()
            except Exception:  # best-effort teardown; never raise from close()
                logger.debug("Chroma client teardown raised; dropping reference", exc_info=True)
            finally:
                self._client = None
                self._col = None

    def __enter__(self) -> ChromaStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _result_to_chunk(cid: str, document: str, metadata: object) -> Chunk:
    """Convert one Chroma query row back into a core :class:`Chunk`.

    The empty-string sentinels stored for ``prev_id``/``next_id`` (Chroma metadata
    cannot hold ``None``) are converted back to ``None`` here.

    Args:
        cid: The chunk id returned by Chroma.
        document: The stored document text for the chunk.
        metadata: The stored metadata mapping for the chunk.

    Returns:
        The reconstructed core :class:`Chunk` (without its embedding, which is not
        requested back from the store).
    """
    md = metadata if isinstance(metadata, dict) else {}
    prev_id = md.get("prev_id") or None
    next_id = md.get("next_id") or None
    return Chunk(
        id=cid,
        doc_id=str(md.get("doc_id", "")),
        position=int(md.get("position", 0)),
        text=document,
        prev_id=str(prev_id) if prev_id else None,
        next_id=str(next_id) if next_id else None,
    )
