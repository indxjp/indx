"""VertexVectorStore — opt-in Vertex AI Vector Search store. Requires ``gcp-vectorsearch``.

Wraps Google Cloud Vertex AI Vector Search (formerly Matching Engine) behind the tiny
:class:`~indx.store.base.VectorStore` Protocol (``name``, ``upsert``, ``search``,
``delete``). The heavy ``google-cloud-aiplatform`` SDK is imported lazily inside
``__init__`` so importing this module is always safe; selecting the slot without the extra
installed raises :class:`~indx.errors.MissingExtraError` with an actionable
``pip install indx[gcp-vectorsearch]`` hint (design-doc §5.3, coding-standards §6).

**Why this is opt-in, not the default GCP store.** Unlike BigQuery (serverless, scales to
zero), Vertex Vector Search **requires a *deployed* index endpoint** before it can answer a
query — a long-running, *always-billed* resource whose deployment takes roughly **20–60
minutes**. Creating the index, deploying it to an endpoint, and supplying
``deployed_index_id`` are operational prerequisites done out-of-band; this adapter only
*references* them. That deploy cost is exactly why this store is gated behind the
``gcp-vectorsearch`` sub-extra and is *not* the frictionless "build then query" default
(design-doc §7.3 / §5.3).

Credentials come from Google Application Default Credentials (ADC) — the SDK's own auth
chain — never from config (coding-standards §6.5). Config sub-tables carry only non-secret
knobs: ``project``, ``location``, ``index_id``, ``index_endpoint_id``, ``deployed_index_id``.

Vendor types never leak past this module: :class:`~indx.core.chunk.Chunk` is converted to a
plain datapoint dict at the edge, and a returned neighbor (id + distance) is converted back
to a core :class:`~indx.store.base.SearchHit`. The cosine *distance* the index returns is
inverted to a "higher is better" score (``1 - distance``) to satisfy the SearchHit contract,
matching how ``qdrant.py`` orders descending.

**Limitation — no stored text.** The Vector Search index stores only the datapoint id and
its feature vector; it does **not** hold the chunk text or payload. So the chunks carried by
returned :class:`SearchHit` objects have their id echoed into both ``id`` and ``doc_id`` and
an empty ``text=""``. Callers that need the original text must resolve it from their own
manifest by id (this is documented and intentional, mirroring the index's capabilities).
"""

from __future__ import annotations

import logging
from typing import Any

from indx.core.chunk import Chunk
from indx.errors import StageError
from indx.store.base import SearchHit
from indx.utils.lazy import require_extra

logger = logging.getLogger(__name__)

# Vertex AI streaming upsert accepts at most ~10,000 datapoints/request; Google recommends
# ~1,000/batch, so we slice large corpora to stay well under the cap.
_BATCH = 1000


class VertexVectorStore:
    """Vector store backed by Vertex AI Vector Search (a *deployed* index endpoint).

    When ``dim`` is supplied (e.g. forwarded by the pipeline from the embedder's real width,
    or pinned via ``[store.vertex-vector] dim``) it is recorded up front. When ``dim`` is
    ``None`` (an SDK user constructing the store directly without knowing the embedder's
    width) sizing is *deferred* until the first :meth:`upsert`, where it is taken from
    ``len(first_vector)`` — so the width always comes from a real vector, never a hard-coded
    guess (T10). Unlike collection-creating stores, sizing here is purely a bookkeeping
    guard: the underlying index is created out-of-band, so ``dim`` exists to make
    :meth:`search` before any upsert fail with a clean typed error rather than a confusing
    vendor one.

    Attributes:
        name: The registry key for this store (``"vertex-vector"``).
    """

    name = "vertex-vector"

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        index_id: str | None = None,
        index_endpoint_id: str | None = None,
        deployed_index_id: str | None = None,
        dim: int | None = None,
    ) -> None:
        """Construct the store; initialize the SDK and record the (out-of-band) resources.

        Args:
            project: GCP project id. Defaults to ADC / ``GOOGLE_CLOUD_PROJECT`` when ``None``.
            location: GCP region the index/endpoint live in (e.g. ``"us-central1"``).
            index_id: Resource id (or full name) of the ``STREAM_UPDATE`` index that
                :meth:`upsert`/:meth:`delete` write to.
            index_endpoint_id: Resource id (or full name) of the *deployed* index endpoint
                that :meth:`search` queries.
            deployed_index_id: The id under which the index is deployed on the endpoint
                (required by ``find_neighbors``).
            dim: Vector dimensionality. When ``None``, recorded lazily from the first
                upserted vector's width.

        Raises:
            MissingExtraError: If the ``gcp-vectorsearch`` extra is not installed.
            StageError: If the Vertex AI SDK cannot be initialized.
        """
        # First statement: fail with one actionable message if the extra is absent, so
        # importing this module stays safe (coding-standards §6.3).
        require_extra("store", "vertex-vector", "gcp-vectorsearch", "google.cloud.aiplatform")
        from google.cloud import (  # type: ignore[import-not-found]  # optional extra: gcp-vectorsearch (lazy)
            aiplatform,
        )

        self._aiplatform = aiplatform
        self._project = project
        self._location = location
        self._index_id = index_id
        self._index_endpoint_id = index_endpoint_id
        self._deployed_index_id = deployed_index_id
        self._dim = dim
        try:
            # Credentials come from ADC inside the SDK — we only pass non-secret project/region.
            aiplatform.init(project=project, location=location)
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("store", f"Vertex Vector Search initialization failed: {exc}") from exc
        logger.debug(
            "initialized Vertex Vector Search store dim=%s index=%s endpoint=%s",
            dim,
            index_id,
            index_endpoint_id,
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Stream-upsert embedded chunks into the index as datapoints.

        Chunks without an ``embedding`` are skipped — a chunk may reach the store without a
        vector if the Embed stage was disabled, and pushing ``None`` into a vector field is
        an error. Only the id and vector are sent; the index does not store chunk text.

        Args:
            chunks: The chunks to store. Each carries its vector and id.

        Raises:
            StageError: If ``index_id`` was not configured, or the upsert call fails.
        """
        if self._index_id is None:
            raise StageError(
                "store",
                "Vertex Vector Search store has no index_id: pass index_id= "
                "(of a STREAM_UPDATE index) before upserting.",
            )
        ready = [c for c in chunks if c.embedding is not None]
        datapoints = [
            # Convert Chunk -> vendor datapoint dict AT THE EDGE (coding-standards §6.2).
            {"datapoint_id": c.id, "feature_vector": list(c.embedding or [])}
            for c in ready
        ]
        if not datapoints:
            return
        try:
            # Deferred sizing: record the width from the first real vector when unknown (T10).
            if self._dim is None:
                self._dim = len(ready[0].embedding or [])
            index = self._aiplatform.MatchingEngineIndex(self._index_id)
            # Slice into ~1,000-datapoint batches so a large corpus does not exceed the
            # streaming upsert per-request cap (mirrors sibling cloud stores).
            for i in range(0, len(datapoints), _BATCH):
                index.upsert_datapoints(datapoints=datapoints[i : i + _BATCH])
        except Exception as exc:
            raise StageError("store", f"Vertex Vector Search upsert failed: {exc}") from exc

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
        """Return the top-``k`` neighbors in descending score order.

        Queries the *deployed* index endpoint. Because the index stores no payload, each
        returned :class:`~indx.store.base.SearchHit` carries a chunk whose ``id``/``doc_id``
        are the neighbor id and whose ``text`` is empty — resolve the original text from your
        own manifest by id.

        Args:
            vector: The query embedding.
            k: Maximum number of neighbors to return.

        Returns:
            Up to ``k`` :class:`~indx.store.base.SearchHit`, highest score first. Ties are
            broken on ``chunk.id`` so ordering is deterministic across runs.

        Raises:
            StageError: If the store was never sized (no ``dim`` and no prior upsert), if the
                endpoint/deployed-index were not configured, or if the query call fails.
        """
        # Searching before the store has been sized (no dim given, nothing upserted yet) is a
        # usage error — surface a clean typed IndxError, not a raw vendor error.
        if self._dim is None:
            raise StageError(
                "store",
                "Vertex Vector Search store has no dimension yet: upsert at least one "
                "vector (or pass dim=) before searching.",
            )
        if self._index_endpoint_id is None or self._deployed_index_id is None:
            raise StageError(
                "store",
                "Vertex Vector Search store cannot search without a deployed endpoint: "
                "pass index_endpoint_id= and deployed_index_id= (the index must be "
                "deployed to an endpoint first, ~20-60 min).",
            )
        try:
            endpoint = self._aiplatform.MatchingEngineIndexEndpoint(self._index_endpoint_id)
            response = endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=[vector],
                num_neighbors=k,
            )
        except Exception as exc:
            raise StageError("store", f"Vertex Vector Search query failed: {exc}") from exc

        # find_neighbors returns one neighbor list per query; we sent a single query.
        neighbors = response[0] if response else []
        hits = [_neighbor_to_hit(n) for n in neighbors]
        # Tie-break on chunk id so equal-score ties resolve identically (coding-standards §1.4).
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete datapoints by chunk id.

        Args:
            chunk_ids: The ids of the chunks (datapoints) to remove.

        Raises:
            StageError: If ``index_id`` was not configured, or the delete call fails.
        """
        if not chunk_ids:
            return
        if self._index_id is None:
            raise StageError(
                "store",
                "Vertex Vector Search store has no index_id: pass index_id= before deleting.",
            )
        try:
            index = self._aiplatform.MatchingEngineIndex(self._index_id)
            index.remove_datapoints(datapoint_ids=list(chunk_ids))
        except Exception as exc:
            raise StageError("store", f"Vertex Vector Search delete failed: {exc}") from exc


def _neighbor_to_hit(neighbor: Any) -> SearchHit:
    """Convert a Vertex Vector Search neighbor back to a core :class:`SearchHit` (edge).

    The index returns only the datapoint id and a cosine ``distance``; there is no stored
    text or payload, so the chunk echoes the id into ``id``/``doc_id`` with empty ``text``.
    Distance is inverted to a higher-is-better ``score`` (``1 - distance``) to satisfy the
    cross-store SearchHit contract.
    """
    nid = str(neighbor.id)
    distance = float(getattr(neighbor, "distance", 0.0))
    chunk = Chunk(id=nid, doc_id=nid, position=0, text="")
    return SearchHit(chunk, 1.0 - distance)
