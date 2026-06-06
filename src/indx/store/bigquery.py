"""BigQueryStore — the default GCP vector store. Requires the ``gcp`` extra.

Persists embedded :class:`~indx.core.chunk.Chunk`s in a single BigQuery table
(``id STRING`` key, ``embedding ARRAY<FLOAT64>``, plus provenance columns) and answers
nearest-neighbor queries via BigQuery's ``VECTOR_SEARCH`` table-valued function with
cosine distance. ``google-cloud-bigquery`` is heavy/optional, so it is imported lazily
inside ``__init__`` and gated by :func:`~indx.utils.lazy.require_extra`; importing this
module is always safe (file-architecture §5, coding-standards §6.3).

BigQuery is serverless with no endpoint to deploy — a fresh store works with zero
index-build wait (small corpora brute-force fine) — which is why it is the *pure-GCP,
zero-infra* default over Vertex Vector Search. The trade-off is seconds-scale query
latency and per-query billing; if the user already runs Postgres, ``pgvector`` is the
lower-latency GCP-friendly alternative (cloud-providers-design §5.3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

_logger = logging.getLogger(__name__)

_TABLE = "indx_chunks"


class BigQueryStore:
    """Vector store backed by BigQuery vector search (serverless, no endpoint).

    Chunks live in one table keyed by ``id``, with the vector in an
    ``embedding ARRAY<FLOAT64>`` column and provenance (``content``/``doc_id``/
    ``position``/``prev_id``/``next_id``) in plain scalar columns. Search uses the
    ``VECTOR_SEARCH`` table-valued function with ``distance_type => 'COSINE'`` and
    converts distance to a "higher is better" similarity score (``1 - distance``) for the
    cross-store contract.

    Credentials come from Google's Application Default Credentials chain (e.g.
    ``GOOGLE_APPLICATION_CREDENTIALS``), never from config; the project may be supplied
    explicitly or read from ``GOOGLE_CLOUD_PROJECT`` (coding-standards §13). Config carries
    only non-secret knobs (``project``/``dataset``/``table``).

    When ``dim`` is supplied (forwarded by the pipeline from the embedder's real width, or
    pinned via ``[store.bigquery] dim``) the table is created up-front. When ``dim`` is
    ``None`` (an SDK user constructing the store directly) table creation is *deferred* to
    the first :meth:`upsert`, where ``dim`` is recorded from the first vector's width — the
    ``ARRAY<FLOAT64>`` column itself is unsized, so ``dim`` only documents the width (T10).

    Attributes:
        name: Stable registry identifier recorded into the archive manifest.
    """

    name = "bigquery"

    def __init__(
        self,
        *,
        project: str | None = None,
        dataset: str = "indx",
        table: str = _TABLE,
        dim: int | None = None,
    ) -> None:
        """Construct the BigQuery client and ensure (or defer) the chunk table.

        Args:
            project: GCP project id. When ``None``, falls back to the
                ``GOOGLE_CLOUD_PROJECT`` environment variable, then the client's own
                default resolution from the ambient credentials.
            dataset: BigQuery dataset that holds the chunk table.
            table: Table name within the dataset.
            dim: Dimensionality of the stored vectors. When supplied the table is created
                up-front; when ``None``, table creation is deferred to the first upsert and
                ``dim`` is recorded from the first vector's width.

        Raises:
            MissingExtraError: If the ``gcp`` extra is not installed.
            StageError: If the client cannot be created or the table cannot be ensured.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("store", "bigquery", "gcp", "google.cloud.bigquery")
        from google.cloud import (  # type: ignore[import-not-found]  # optional extra: gcp
            bigquery,
        )

        self._bigquery = bigquery
        self.dim = dim
        # Project resolution: explicit config wins, else the standard GCP env var; the
        # client resolves credentials from the ADC chain (never read here).
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._dataset = dataset
        self._table = table
        try:
            self._client = bigquery.Client(project=project)
            # Fully-qualified id ``project.dataset.table`` for SQL. The client's resolved
            # project is the source of truth so it matches the credentials in use.
            self._table_id = f"{self._client.project}.{dataset}.{table}"
            # Pre-size only when a dim is known; otherwise defer to the first upsert so the
            # table is created once a real vector's width is in hand (T10 lazy sizing).
            if dim is not None:
                self._ensure_table(dim)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"BigQuery initialization failed: {exc}") from exc
        _logger.debug("initialized BigQuery store table=%s dim=%s", self._table_id, dim)

    def _ensure_table(self, dim: int) -> None:
        """Create the chunk table if absent and record the resolved ``dim``.

        Idempotent (``exists_ok=True``) and records ``dim`` so a single sizing path serves
        both up-front (``dim`` given) and lazy (first-upsert) construction. The
        ``embedding`` column is ``ARRAY<FLOAT64>`` — its width is the vector's own, so
        ``dim`` only documents it (T10).
        """
        bq = self._bigquery
        schema = [
            bq.SchemaField("id", "STRING"),
            bq.SchemaField("content", "STRING"),
            bq.SchemaField("doc_id", "STRING"),
            bq.SchemaField("position", "INT64"),
            bq.SchemaField("prev_id", "STRING"),
            bq.SchemaField("next_id", "STRING"),
            bq.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        table = bq.Table(self._table_id, schema=schema)
        self._client.create_table(table, exists_ok=True)
        self.dim = dim

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunks, keyed by ``id``, via a parameterized ``MERGE``.

        Un-embedded chunks (``embedding is None``) are skipped: an un-embedded chunk has no
        vector to store and is not searchable anyway (reference §3).

        Args:
            chunks: Chunks to persist. May be empty (a no-op).

        Raises:
            StageError: If the write fails.
        """
        bq = self._bigquery
        ready = [c for c in chunks if c.embedding is not None]
        if not ready:
            return
        try:
            # Deferred sizing: if the table was never created (dim unknown at __init__),
            # create it now and record the first real vector's width (T10).
            if self.dim is None:
                self._ensure_table(len(ready[0].embedding or []))
            # One parameterized MERGE per chunk: matched rows are updated, unmatched are
            # inserted — upsert semantics keyed by id.
            for chunk in ready:
                sql = (
                    f"MERGE `{self._table_id}` AS t "
                    f"USING (SELECT @id AS id) AS s ON t.id = s.id "
                    f"WHEN MATCHED THEN UPDATE SET "
                    f"content = @content, doc_id = @doc_id, position = @position, "
                    f"prev_id = @prev_id, next_id = @next_id, embedding = @embedding "
                    f"WHEN NOT MATCHED THEN INSERT "
                    f"(id, content, doc_id, position, prev_id, next_id, embedding) "
                    f"VALUES (@id, @content, @doc_id, @position, @prev_id, @next_id, @embedding)"
                )
                job_config = bq.QueryJobConfig(
                    query_parameters=[
                        bq.ScalarQueryParameter("id", "STRING", chunk.id),
                        bq.ScalarQueryParameter("content", "STRING", chunk.text),
                        bq.ScalarQueryParameter("doc_id", "STRING", chunk.doc_id),
                        bq.ScalarQueryParameter("position", "INT64", chunk.position),
                        bq.ScalarQueryParameter("prev_id", "STRING", chunk.prev_id),
                        bq.ScalarQueryParameter("next_id", "STRING", chunk.next_id),
                        bq.ArrayQueryParameter("embedding", "FLOAT64", chunk.embedding),
                    ]
                )
                self._client.query(sql, job_config=job_config).result()
        except Exception as exc:
            raise StageError("store", f"BigQuery upsert failed: {exc}") from exc

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
        # Searching before the table exists (no dim given, nothing upserted yet) is a usage
        # error — surface a clean typed IndxError, not a raw "table not found".
        if self.dim is None:
            raise StageError(
                "store",
                "BigQuery store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        bq = self._bigquery
        # VECTOR_SEARCH returns ``distance`` (cosine, 0 = identical); return
        # ``1 - distance`` so the score is "higher is better" per the SearchHit contract.
        sql = (
            f"SELECT base.id AS id, base.content AS content, base.doc_id AS doc_id, "
            f"base.position AS position, base.prev_id AS prev_id, base.next_id AS next_id, "
            f"1 - distance AS score "
            f"FROM VECTOR_SEARCH("
            f"TABLE `{self._table_id}`, 'embedding', "
            f"(SELECT @q AS embedding), top_k => @k, distance_type => 'COSINE')"
        )
        job_config = bq.QueryJobConfig(
            query_parameters=[
                bq.ArrayQueryParameter("q", "FLOAT64", vector),
                bq.ScalarQueryParameter("k", "INT64", k),
            ]
        )
        try:
            rows = list(self._client.query(sql, job_config=job_config).result())
        except Exception as exc:
            raise StageError("store", f"BigQuery query failed: {exc}") from exc
        hits = [SearchHit(_row_to_chunk(row), float(row["score"])) for row in rows]
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
        bq = self._bigquery
        sql = f"DELETE FROM `{self._table_id}` WHERE id IN UNNEST(@ids)"
        job_config = bq.QueryJobConfig(
            query_parameters=[bq.ArrayQueryParameter("ids", "STRING", list(chunk_ids))]
        )
        try:
            self._client.query(sql, job_config=job_config).result()
        except Exception as exc:
            raise StageError("store", f"BigQuery delete failed: {exc}") from exc


def _row_to_chunk(row: Any) -> Chunk:
    """Convert a BigQuery result row back to a core :class:`Chunk` (vendor edge)."""
    return Chunk(
        id=str(row["id"]),
        doc_id=row["doc_id"],
        position=row["position"],
        text=row["content"],
        prev_id=row["prev_id"],
        next_id=row["next_id"],
    )
