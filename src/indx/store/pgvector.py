"""PgVectorStore — Postgres + pgvector vector store. Requires the ``pgvector`` extra.

Persists embedded :class:`~indx.core.chunk.Chunk`s in a single ``indx_chunks`` table
(``id`` PK, ``embedding vector``, ``payload jsonb``) and answers nearest-neighbor queries
via cosine distance. ``psycopg`` (v3) and ``pgvector`` are heavy/optional, so they are
imported lazily inside ``__init__`` and gated by :func:`~indx.utils.lazy.require_extra`;
importing this module is always safe (file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

_TABLE = "indx_chunks"
_BATCH = 256


class PgVectorStore:
    """Vector store backed by Postgres + the ``pgvector`` extension.

    Chunks live in one table keyed by ``id``, with the vector in an ``embedding``
    column and provenance (``doc_id``/``position``/``text``/``prev_id``/``next_id``) in a
    ``payload`` ``jsonb`` column. Search orders by cosine distance (``<=>``) and converts
    distance to a "higher is better" similarity score for the cross-store contract.

    The connection string (DSN) is supplied by the caller and never logged; secrets are
    read from the environment by the user, not hard-coded (coding-standards §13).

    Attributes:
        name: Stable registry identifier recorded into the archive manifest.
    """

    name = "pgvector"

    def __init__(self, dsn: str, *, dim: int | None = None) -> None:
        """Connect to Postgres and ensure (or defer) the chunk table.

        When ``dim`` is supplied (forwarded by the pipeline from the embedder's real width,
        or pinned via ``[store.pgvector] dim``) the ``vector`` column is created sized to
        ``dim`` up-front. When ``dim`` is ``None`` (an SDK user constructing the store
        directly) table creation is *deferred* to the first :meth:`upsert`, where the
        column is sized to the width of the first vector — never a hard-coded guess (T10).

        Args:
            dsn: psycopg connection string for the target database.
            dim: Dimensionality of the stored vectors. Must match the embedder. When
                ``None``, the table is created lazily on the first upsert.

        Raises:
            MissingExtraError: If the ``pgvector`` extra is not installed.
            StageError: If the connection or table setup fails.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("store", "pgvector", "pgvector", "psycopg")
        import psycopg  # type: ignore[import-not-found]  # optional extra: pgvector
        from pgvector.psycopg import (  # type: ignore[import-not-found]  # optional extra: pgvector
            register_vector,
        )

        self.dim = dim
        # Set before the try so a failure mid-setup still leaves a well-defined attr
        # and the except block can release a half-open connection (no orphaned socket).
        self._conn: Any = None
        try:
            self._conn = psycopg.connect(dsn)
            # pgvector requires the extension to exist and register_vector to be called
            # on every connection before vector params can adapt (reference §5).
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
            # Pre-size the table only when a dim is known; otherwise defer to the first
            # upsert so the vector column is sized to a real vector's width (T10).
            if dim is not None:
                self._ensure_table(dim)
            self._conn.commit()
        except Exception as exc:  # normalize backend errors to a typed IndxError
            # Close the connection opened above before re-raising; __init__ raising leaves
            # a half-built object Python never finalizes, so the socket would leak.
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
            raise StageError("store", str(exc)) from exc

    def _ensure_table(self, dim: int) -> None:
        """Create the chunk table with the ``vector`` column sized to ``dim`` if absent.

        Idempotent (``CREATE TABLE IF NOT EXISTS``) and records the resolved ``dim`` so a
        single sizing path serves both up-front (``dim`` given) and lazy (first-upsert)
        construction. The caller owns the surrounding transaction/commit.
        """
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            f"  id text PRIMARY KEY,"
            f"  embedding vector({dim}),"
            f"  payload jsonb)"
        )
        self.dim = dim

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks, keyed by ``id``.

        Un-embedded chunks (``embedding is None``) are skipped: a backend column typed
        ``vector`` cannot accept ``NULL`` vectors here, and an un-embedded chunk is not
        searchable anyway (reference §3).

        Args:
            chunks: Chunks to persist. May be empty.

        Raises:
            StageError: If the write fails.
        """
        from psycopg.types.json import (  # type: ignore[import-not-found]  # optional extra: pgvector
            Jsonb,
        )

        ready = [c for c in chunks if c.embedding is not None]
        rows = [(c.id, c.embedding, Jsonb(_chunk_to_payload(c))) for c in ready]
        if not rows:
            return
        try:
            # Deferred sizing: if the table was never created (dim unknown at __init__),
            # size its vector column now to the first real vector's width (T10).
            if self.dim is None:
                self._ensure_table(len(ready[0].embedding or []))
                self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise StageError("store", str(exc)) from exc
        sql = (
            f"INSERT INTO {_TABLE} (id, embedding, payload) VALUES (%s, %s, %s) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"embedding = EXCLUDED.embedding, payload = EXCLUDED.payload"
        )
        try:
            # Batch writes rather than one round-trip per chunk (coding-standards §14).
            for start in range(0, len(rows), _BATCH):
                self._conn.executemany(sql, rows[start : start + _BATCH])
            self._conn.commit()
        except Exception as exc:
            # A failed statement aborts the psycopg3 tx; roll back so this connection
            # is reusable rather than stuck "current transaction is aborted".
            self._conn.rollback()
            raise StageError("store", str(exc)) from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending similarity order.

        Args:
            vector: Query embedding.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`SearchHit`s, highest similarity first. Ties are broken on
            ``chunk.id`` so ordering is deterministic across runs (coding-standards §1.4).

        Raises:
            StageError: If the store was never sized (no ``dim`` and no prior upsert), or
                if the query fails.
        """
        # Searching before the table exists (no dim given, nothing upserted yet) is a
        # usage error — surface a clean typed IndxError, not a raw "relation does not exist".
        if self.dim is None:
            raise StageError(
                "store",
                "pgvector store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        # ``<=>`` is cosine DISTANCE (0 = identical); ORDER BY it ascending for nearest
        # neighbors, and return ``1 - distance`` so the score is "higher is better".
        sql = (
            f"SELECT id, payload, 1 - (embedding <=> %s) AS score "
            f"FROM {_TABLE} ORDER BY embedding <=> %s LIMIT %s"
        )
        try:
            rows = self._conn.execute(sql, (vector, vector, k)).fetchall()
        except Exception as exc:
            raise StageError("store", str(exc)) from exc
        hits = [SearchHit(_row_to_chunk(row), float(row[2])) for row in rows]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ``id``.

        Args:
            chunk_ids: Ids to remove. May be empty (a no-op).

        Raises:
            StageError: If the delete fails.
        """
        if not chunk_ids:
            return
        try:
            self._conn.execute(
                f"DELETE FROM {_TABLE} WHERE id = ANY(%s)",
                (list(chunk_ids),),
            )
            self._conn.commit()
        except Exception as exc:
            # A failed statement aborts the psycopg3 tx; roll back so this connection
            # is reusable rather than stuck "current transaction is aborted".
            self._conn.rollback()
            raise StageError("store", str(exc)) from exc

    def close(self) -> None:
        """Release the underlying Postgres connection.

        Idempotent: safe to call more than once, and on a store whose ``__init__``
        already released the connection on a setup failure. After ``close()`` the store
        must not be used again.
        """
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Convert a core ``Chunk`` to a JSON payload (vendor edge, coding-standards §6.2)."""
    return {
        "doc_id": chunk.doc_id,
        "position": chunk.position,
        "text": chunk.text,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
    }


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    """Convert a ``(id, payload, score)`` row back to a core ``Chunk`` (vendor edge)."""
    payload = row[1] or {}
    return Chunk(
        id=str(row[0]),
        doc_id=payload["doc_id"],
        position=payload["position"],
        text=payload["text"],
        prev_id=payload.get("prev_id"),
        next_id=payload.get("next_id"),
    )
