"""S3VectorsStore — the default AWS vector store. Requires the ``aws`` extra.

Wraps the boto3 ``s3vectors`` client behind the tiny :class:`~indx.store.base.VectorStore`
Protocol (``name``, ``upsert``, ``search``, ``delete``). The heavy ``boto3`` SDK is imported
lazily inside ``__init__`` so importing this module is always safe; selecting the slot without
the extra installed raises :class:`~indx.errors.MissingExtraError` with an actionable
``pip install indx[aws]`` hint (design §5.1, coding-standards §6).

Amazon S3 Vectors is a serverless, zero-infrastructure vector store: a *vector bucket* holds
named *vector indexes*, and ``put_vectors``/``query_vectors``/``delete_vectors`` map 1:1 to our
upsert/search/delete contract. Credentials are never touched here — ``boto3`` resolves them via
its standard chain (env / shared profile / SSO / role); the adapter supplies only ``region_name``
(coding-standards §13). Vendor types never leak past this module: ``Chunk`` is converted to/from a
vector record only at the edge (coding-standards §6.2). S3 Vectors returns cosine *distance*
(0 = identical), which is inverted to a "higher is better" ``score`` for the SearchHit contract.

The ``s3vectors`` client first ships in **boto3 ≥ 1.40**; an older boto3 cannot create it, so the
adapter raises a clear :class:`~indx.errors.StageError` telling the user to upgrade (design §12.2).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

logger = logging.getLogger(__name__)

_DEFAULT_INDEX = "indx"
_METRIC = "cosine"

# AWS S3 Vectors hard per-request caps: PutVectors and DeleteVectors accept at most 500
# vectors/keys per call, and QueryVectors caps topK at 100. Large corpora must be sliced into
# batches or the real service rejects the call (the offline fake does not enforce these caps).
_PUT_BATCH = 500
_DELETE_BATCH = 500
_MAX_TOPK = 100


class S3VectorsStore:
    """Vector store backed by Amazon S3 Vectors (serverless vector bucket + index).

    When ``dim`` is supplied (e.g. forwarded by the pipeline from the embedder's real width, or
    pinned via ``[store.s3vectors] dim``) the vector index is sized to ``dim`` on the first
    :meth:`upsert`. When ``dim`` is ``None`` (an SDK user constructing the store directly without
    knowing the embedder's width) the index is sized to ``len(first_vector)`` at the first upsert
    — so the width always comes from a real vector, never a hard-coded guess (T10). The index is
    created lazily on first write if it does not yet exist. Vector keys are the
    :class:`~indx.core.chunk.Chunk` ids (strings pass straight through); the remaining chunk
    fields ride along as the vector's metadata.

    The vector ``bucket`` is required for any real use: S3 Vectors stores vectors *inside* a
    named bucket, so :meth:`upsert`/:meth:`search`/:meth:`delete` raise a clean
    :class:`~indx.errors.StageError` when no bucket is configured.

    Attributes:
        name: The registry key for this store (``"s3vectors"``).
    """

    name = "s3vectors"

    def __init__(
        self,
        *,
        bucket: str | None = None,
        index: str = _DEFAULT_INDEX,
        region: str | None = None,
        metric: str = _METRIC,
        dim: int | None = None,
    ) -> None:
        """Construct the store and the boto3 ``s3vectors`` client.

        Args:
            bucket: Name of the S3 vector bucket holding the index. Required for real use; when
                omitted, the store's data methods raise :class:`StageError`.
            index: Name of the vector index within the bucket. Defaults to ``"indx"``.
            region: AWS region (e.g. ``"us-east-1"``). When ``None``, boto3 resolves it from
                ``AWS_REGION``/``AWS_DEFAULT_REGION`` or the active profile.
            metric: Distance metric for the index when it is created. Defaults to ``"cosine"``.
            dim: Vector dimensionality the index is sized to. When ``None``, the index is sized
                on the first :meth:`upsert` to the width of the first vector.

        Raises:
            MissingExtraError: If the ``aws`` extra (``boto3``) is not installed.
            StageError: If boto3 is too old to provide the ``s3vectors`` client, or the client
                cannot be created.
        """
        # First statement: fail with one actionable message if the extra is absent, so importing
        # this module stays safe (coding-standards §6.3).
        require_extra("store", "s3vectors", "aws", "boto3")
        import boto3  # type: ignore[import-not-found]  # optional extra: aws (lazy)

        self._bucket = bucket
        self._index = index
        self._metric = metric
        self._dim = dim
        self._index_ready = False
        try:
            # boto3 < 1.40 has no s3vectors service model and raises UnknownServiceError here.
            self._client = boto3.client("s3vectors", region_name=region)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError(
                "store",
                "S3 Vectors requires boto3>=1.40 (the 's3vectors' client is unavailable): "
                f"upgrade boto3 to use this store. Underlying error: {exc}",
            ) from exc
        logger.debug("initialized S3 Vectors store bucket=%s index=%s dim=%s", bucket, index, dim)

    def _require_bucket(self) -> str:
        """Return the configured bucket or raise a clean typed error."""
        if not self._bucket:
            raise StageError(
                "store",
                "S3 Vectors store has no bucket: pass bucket= (the vector bucket name) "
                "before upserting, searching, or deleting.",
            )
        return self._bucket

    def _ensure_index(self, dim: int) -> None:
        """Create the vector index sized to ``dim`` if it does not already exist.

        Idempotent and records the resolved ``dim`` so later upserts reuse it rather than
        re-checking width. The boto3 method is ``create_index`` (NOT ``create_vector_index``)
        and requires ``dataType`` alongside ``dimension``/``distanceMetric``. S3 Vectors
        raises a ``ConflictException`` when the index already exists, which is treated as
        success; any *other* error is surfaced rather than masked as "already exists" (a
        wrong method name or bad params must not be silently swallowed).
        """
        bucket = self._require_bucket()
        try:
            self._client.create_index(
                vectorBucketName=bucket,
                indexName=self._index,
                dataType="float32",
                dimension=dim,
                distanceMetric=self._metric,
            )
        except Exception as exc:  # noqa: BLE001 — only a pre-existing index is benign
            already = "Conflict" in type(exc).__name__ or "already exist" in str(exc).lower()
            if not already:
                raise StageError("store", f"S3 Vectors create_index failed: {exc}") from exc
            logger.debug("create_index: index already exists: %s", exc)
        self._dim = dim
        self._index_ready = True

    def upsert(self, chunks: list[Chunk]) -> None:
        """Upsert embedded chunks into the vector index.

        Chunks without an ``embedding`` are skipped — a chunk may reach the store without a
        vector if the Embed stage was disabled, and pushing ``None`` into a vector field is an
        error. ``put_vectors`` is an upsert: a repeated key overwrites the prior vector.

        Args:
            chunks: The chunks to store. Each carries its vector and provenance.

        Raises:
            StageError: If no bucket is configured, or the S3 Vectors call fails.
        """
        bucket = self._require_bucket()
        ready = [c for c in chunks if c.embedding is not None]
        if not ready:
            return
        try:
            # Deferred sizing: if the index was never sized at construction (dim was unknown),
            # size it now to the first real vector's width — not a guess (T10).
            if self._dim is None or not self._index_ready:
                self._ensure_index(self._dim or len(ready[0].embedding or []))
            vectors = [
                {
                    "key": c.id,
                    "data": {"float32": list(c.embedding or [])},
                    "metadata": _chunk_to_metadata(c),  # convert Chunk -> metadata AT THE EDGE
                }
                for c in ready
            ]
            # AWS caps PutVectors at 500 vectors/call; slice so the per-batch retry/backoff
            # in _put_vectors applies to each batch.
            for i in range(0, len(vectors), _PUT_BATCH):
                self._put_vectors(bucket, vectors[i : i + _PUT_BATCH])
        except StageError:
            raise
        except Exception as exc:
            raise StageError("store", f"S3 Vectors upsert failed: {exc}") from exc

    def _put_vectors(self, bucket: str, vectors: list[dict[str, Any]]) -> None:
        """Write vectors, tolerating the brief window before a new index becomes writable.

        ``create_vector_index`` is eventually consistent on S3 Vectors: a ``put_vectors``
        issued right after the index is created can race its propagation and fail with a
        ``NotFoundException`` ("The specified index could not be found"). Retry with bounded
        exponential backoff — re-ensuring the index each round — before surfacing the error,
        so the lazy create-then-write path on a fresh bucket works without a manual wait.
        """
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                self._client.put_vectors(
                    vectorBucketName=bucket, indexName=self._index, vectors=vectors
                )
                return
            except Exception as exc:  # noqa: BLE001 — inspect, then retry-or-reraise
                not_found = "NotFound" in type(exc).__name__ or "not be found" in str(exc)
                if not not_found:
                    raise
                last_exc = exc
                logger.debug("put_vectors retry %d (index not ready yet): %s", attempt, exc)
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                if self._dim is not None:  # nudge the index into existence again
                    self._ensure_index(self._dim)
        if last_exc is not None:  # retries exhausted — let upsert wrap it as StageError
            raise last_exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` hits in descending score order.

        Args:
            vector: The query embedding.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit`, highest score first (cosine
            ``score = 1 - distance``). Ties are broken on ``chunk.id`` so ordering is
            deterministic across runs (coding-standards §1.4). Note: AWS QueryVectors hard-caps
            ``topK`` at 100 and offers no pagination, so a request for ``k > 100`` returns at most
            100 hits; the truncation is logged at WARNING so it is never silent.

        Raises:
            StageError: If no bucket is configured, the store was never sized (no ``dim`` and no
                prior upsert), or the S3 Vectors query call fails.
        """
        bucket = self._require_bucket()
        # Searching before the index exists (no dim given, nothing upserted yet) is a usage
        # error — surface a clean typed IndxError, not a raw vendor "not found".
        if self._dim is None:
            raise StageError(
                "store",
                "S3 Vectors store has no dimension yet: upsert at least one vector "
                "(or pass dim=) before searching.",
            )
        # AWS QueryVectors hard-caps topK at 100 with no pagination, so we cannot honor a larger
        # k no matter what. Don't shortchange the caller silently — warn so the cap is visible.
        top_k = k
        if k > _MAX_TOPK:
            logger.warning(
                "S3 Vectors caps QueryVectors topK at %d; requested k=%d truncated to %d "
                "(this backend cannot return more — no pagination).",
                _MAX_TOPK,
                k,
                _MAX_TOPK,
            )
            top_k = _MAX_TOPK
        try:
            resp = self._client.query_vectors(
                vectorBucketName=bucket,
                indexName=self._index,
                queryVector={"float32": list(vector)},
                topK=top_k,  # AWS caps QueryVectors topK at 100 (see warning above)
                returnDistance=True,
                returnMetadata=True,
            )
        except Exception as exc:
            raise StageError("store", f"S3 Vectors query failed: {exc}") from exc

        hits = [_result_to_chunk_hit(r) for r in resp.get("vectors", [])]
        # Tie-break on chunk id so equal-score ties resolve identically (coding-standards §1.4).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete vectors by chunk id (key).

        Args:
            chunk_ids: The ids of the chunks (keys) to remove. May be empty (a no-op).

        Raises:
            StageError: If no bucket is configured, or the S3 Vectors delete call fails.
        """
        if not chunk_ids:
            return
        bucket = self._require_bucket()
        try:
            # AWS caps DeleteVectors at 500 keys/call; slice into batches.
            ids = list(chunk_ids)
            for i in range(0, len(ids), _DELETE_BATCH):
                self._client.delete_vectors(
                    vectorBucketName=bucket, indexName=self._index, keys=ids[i : i + _DELETE_BATCH]
                )
        except Exception as exc:
            raise StageError("store", f"S3 Vectors delete failed: {exc}") from exc

    def close(self) -> None:
        """Release the underlying boto3 ``s3vectors`` client and its connection pool.

        A botocore client owns an HTTP connection pool; in a long-lived process (e.g.
        ``indx app``) a per-build store whose client is never closed leaks those pooled
        sockets until GC. Best-effort: call the vendor's own ``close()`` if present, then
        drop the reference. Idempotent — safe to call more than once. After ``close()`` the
        store must not be used again.
        """
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                closer = getattr(client, "close", None)
                if callable(closer):
                    closer()
            except Exception:  # best-effort teardown; never raise from close()
                logger.debug("S3 Vectors client teardown raised; dropping reference", exc_info=True)
            finally:
                self._client = None


def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Convert a core :class:`Chunk` to an S3 Vectors metadata dict (vendor edge).

    S3 Vectors rejects ``null`` metadata values (they must be strings, numbers, booleans,
    or arrays), so boundary chunks whose ``prev_id``/``next_id`` are ``None`` omit those
    keys entirely; :func:`_result_to_chunk_hit` reads them back with ``.get`` → ``None``.
    """
    metadata: dict[str, Any] = {
        "source_text": chunk.text,
        "doc_id": chunk.doc_id,
        "position": chunk.position,
    }
    if chunk.prev_id is not None:
        metadata["prev_id"] = chunk.prev_id
    if chunk.next_id is not None:
        metadata["next_id"] = chunk.next_id
    return metadata


def _result_to_chunk_hit(result: dict[str, Any]) -> SearchHit:
    """Convert an S3 Vectors query result to a core :class:`SearchHit` (vendor edge)."""
    metadata = result.get("metadata") or {}
    # Cosine distance (0 = identical) -> "higher is better" similarity for the SearchHit contract.
    score = 1.0 - float(result.get("distance", 0.0))
    chunk = Chunk(
        id=str(result["key"]),
        doc_id=metadata["doc_id"],
        position=metadata["position"],
        text=metadata["source_text"],
        prev_id=metadata.get("prev_id"),
        next_id=metadata.get("next_id"),
    )
    return SearchHit(chunk, score)
