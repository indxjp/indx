"""LanceDBStore — file-based, columnar vector store. Requires the ``lancedb`` extra.

LanceDB is a zero-server, on-disk vector database, well suited to large vector data that
should travel as plain files. It sits behind the tiny :class:`~indx.store.base.VectorStore`
Protocol like every other store. The ``lancedb`` SDK is heavy/optional, so it is imported
lazily inside ``__init__`` and gated by :func:`~indx.utils.lazy.require_extra`; importing
this module is always safe (file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from collections.abc import Mapping

_TABLE = "indx"
_METRIC = "cosine"


class LanceDBStore:
    """Vector store backed by a local LanceDB table.

    A single table named ``indx`` holds rows of ``{id, vector, payload}``: ``id`` is the
    :class:`~indx.core.chunk.Chunk` id, ``vector`` is its embedding, and ``payload`` is the
    JSON-encoded remainder of the chunk so it can be reconstructed at the edge. Searches use
    cosine distance and are normalized to "higher is better" similarity to honor the
    cross-store contract.

    Attributes:
        name: Registry key recorded into the archive manifest.
    """

    name = "lancedb"

    def __init__(self, *, path: str = "./space.lancedb", dim: int | None = None) -> None:
        """Open (or create) the LanceDB table at ``path``.

        When ``dim`` is supplied (forwarded by the pipeline from the embedder's real width,
        or pinned via ``[store.lancedb] dim``) an empty table is seeded with that fixed
        vector width up-front. When ``dim`` is ``None`` (an SDK user constructing the store
        directly) and the table does not yet exist, table creation is *deferred* to the
        first :meth:`upsert`, where it is sized to the width of the first vector — never a
        hard-coded guess (T10). An already-existing table is always opened as-is.

        Args:
            path: On-disk directory for the LanceDB dataset. Created if absent.
            dim: Vector dimensionality, used to seed an empty table's schema. When ``None``
                and no table exists yet, the table is created lazily on the first upsert.

        Raises:
            MissingExtraError: If the ``lancedb`` extra is not installed.
            StageError: If the connection or table cannot be opened.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("store", "lancedb", "lancedb", "lancedb")
        import lancedb  # type: ignore[import-not-found]  # optional extra: lancedb

        self._dim = dim
        # ``None`` until a table is opened/created; lazy upsert checks it before writing.
        self._table: Any = None
        try:
            self._db = lancedb.connect(path)
            if _TABLE in set(self._db.table_names()):
                self._table = self._db.open_table(_TABLE)
            elif dim is not None:
                # Seed the schema sized to the known dim so the vector column has a fixed
                # width before any real upsert. With no dim, defer to the first upsert.
                self._ensure_table(dim)
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("store", str(exc)) from exc

    def _ensure_table(self, dim: int) -> None:
        """Create the table seeded to a fixed ``dim`` vector width if not already open.

        Seeds one placeholder row to fix the vector column width, then clears it. Records
        the resolved ``dim`` so a single sizing path serves both up-front (``dim`` given)
        and lazy (first-upsert) construction.
        """
        if self._table is None:
            seed = [{"id": "", "vector": [0.0] * dim, "payload": "{}"}]
            self._table = self._db.create_table(_TABLE, data=seed, mode="overwrite")
            self._table.delete("id = ''")
        self._dim = dim

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks in the table.

        Chunks without an embedding are skipped (a vector store has nothing to index for
        them). Existing rows with the same id are deleted first so the write is a true
        upsert rather than an append.

        Args:
            chunks: Chunks to persist. May be empty.

        Raises:
            StageError: If the LanceDB write fails.
        """
        ready = [c for c in chunks if c.embedding is not None]
        if not ready:
            return
        rows: list[dict[str, Any]] = [
            {"id": c.id, "vector": list(c.embedding or []), "payload": _chunk_to_payload(c)}
            for c in ready
        ]
        try:
            # Deferred sizing: if the table was never created (dim unknown at __init__),
            # size it now to the first real vector's width — not a guess (T10).
            if self._table is None:
                self._ensure_table(len(ready[0].embedding or []))
            self._table.delete(_id_in_predicate([c.id for c in ready]))
            self._table.add(rows)
        except Exception as exc:
            raise StageError("store", str(exc)) from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending similarity order.

        LanceDB returns a cosine ``_distance`` in ``[0, 2]`` (0 == identical); it is
        converted to a similarity (``1 - distance``) so output matches the cross-store
        "higher is better" contract. Ties are broken on chunk id for determinism
        (coding-standards §1.4).

        Args:
            vector: Query embedding.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit`s, highest score first.

        Raises:
            StageError: If the store was never sized (no ``dim`` and no prior upsert), or
                if the LanceDB query fails.
        """
        # Searching before the table exists (no dim given, nothing upserted yet) is a
        # usage error — surface a clean typed IndxError, not a raw vendor error.
        if self._table is None:
            raise StageError(
                "store",
                "LanceDB store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        try:
            results: list[Mapping[str, Any]] = (
                self._table.search(vector).metric(_METRIC).limit(k).to_list()
            )
        except Exception as exc:
            raise StageError("store", str(exc)) from exc
        hits = [SearchHit(_row_to_chunk(row), 1.0 - float(row["_distance"])) for row in results]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by id.

        Args:
            chunk_ids: Chunk ids to remove. May be empty (a no-op).

        Raises:
            StageError: If the LanceDB delete fails.
        """
        if not chunk_ids:
            return
        try:
            self._table.delete(_id_in_predicate(chunk_ids))
        except Exception as exc:
            raise StageError("store", str(exc)) from exc


def _chunk_to_payload(chunk: Chunk) -> str:
    """Encode a chunk's non-vector fields as a stable JSON string (core -> storage)."""
    return json.dumps(
        {
            "doc_id": chunk.doc_id,
            "position": chunk.position,
            "text": chunk.text,
            "prev_id": chunk.prev_id,
            "next_id": chunk.next_id,
        },
        sort_keys=True,
    )


def _row_to_chunk(row: Mapping[str, Any]) -> Chunk:
    """Reconstruct a core :class:`Chunk` from a stored row (storage -> core, at the edge)."""
    payload: dict[str, Any] = json.loads(row["payload"])
    raw_vector = row.get("vector")
    return Chunk(
        id=str(row["id"]),
        doc_id=payload["doc_id"],
        position=payload["position"],
        text=payload["text"],
        prev_id=payload.get("prev_id"),
        next_id=payload.get("next_id"),
        embedding=[float(x) for x in raw_vector] if raw_vector is not None else None,
    )


def _id_in_predicate(ids: list[str]) -> str:
    """Build a LanceDB SQL ``id IN (...)`` predicate with quoted, escaped ids."""
    quoted = ", ".join("'" + cid.replace("'", "''") + "'" for cid in ids)
    return f"id IN ({quoted})"
