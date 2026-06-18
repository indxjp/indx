"""VertexEmbedder — GCP embedder (``vertex:gemini-embedding-001``).

Wraps Vertex AI's ``gemini-embedding-001`` model behind the tiny
:class:`~indx.embed.base.Embedder` Protocol via the unified ``google-genai`` SDK. The
SDK is heavy/optional, so it is imported lazily inside ``__init__`` and gated by
:func:`~indx.utils.lazy.require_extra`; importing this module is always safe
(cloud-providers-design §5.3, coding-standards §6.3).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from google import genai  # type: ignore[import-not-found]  # optional extra: gcp

# Output dimensions supported by gemini-embedding-001 via ``output_dimensionality``.
# The model's native default is 3072; we pin 768 so it drops into 768-dim stores.
_SUPPORTED_DIMS: frozenset[int] = frozenset({768, 1536, 3072})

_DEFAULT_MODEL = "gemini-embedding-001"
_DEFAULT_DIM = 768
_DEFAULT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
# Per-request size for non-gemini-embedding models (text-embedding-* batch up to ~250).
# gemini-embedding-* is forced to 1 in ``embed`` regardless — Vertex rejects multi-text
# requests for that family.
_DEFAULT_BATCH_SIZE = 32


class VertexEmbedder:
    """Embedder backed by Vertex AI ``gemini-embedding-001`` (via ``google-genai``).

    The default model is ``gemini-embedding-001`` truncated to 768 dimensions through
    ``output_dimensionality`` (the model's native width is 3072) so it drops into 768-dim
    stores; ``dim`` is configurable to 1536/3072. Credentials/project come from the SDK's
    own auth chain and ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``; nothing is
    logged or stored on this object.

    Attributes:
        name: Stable model identifier recorded into the archive manifest.
        dim: Output dimensionality requested via ``output_dimensionality``.
    """

    name = "vertex"
    dim = _DEFAULT_DIM

    def __init__(
        self,
        model: str | None = None,
        *,
        dim: int = _DEFAULT_DIM,
        task_type: str = _DEFAULT_TASK_TYPE,
        project: str | None = None,
        location: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """Construct the embedder and its Vertex genai client.

        Args:
            model: Vertex embedding model id. Defaults to ``gemini-embedding-001``.
            dim: Output dimensionality (768, 1536, or 3072). Defaults to 768.
            task_type: Embedding task hint — ``RETRIEVAL_DOCUMENT`` on upsert,
                ``RETRIEVAL_QUERY`` on search, for best recall.
            project: GCP project id; falls back to the SDK's auth chain / env.
            location: Vertex region; falls back to the SDK's auth chain / env.
            batch_size: Number of texts sent per ``embed_content`` request.

        Raises:
            MissingExtraError: If the ``gcp`` extra is not installed.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("embedder", "vertex", "gcp", "google.genai")
        from google import genai  # optional extra: gcp (lazy; see TYPE_CHECKING import)
        from google.genai import types  # type: ignore[import-not-found]  # optional extra: gcp

        self.model = model or _DEFAULT_MODEL
        self.name = self.model  # record the *actual* model in the manifest
        if dim not in _SUPPORTED_DIMS:
            raise StageError(
                "embed",
                f"gemini-embedding-001 supports {sorted(_SUPPORTED_DIMS)}, not {dim}.",
            )
        self.dim = dim
        self.task_type = task_type
        self.batch_size = batch_size
        self._types = types
        # The SDK resolves credentials/project/location from its own auth chain and the
        # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION env vars; nothing is stored here.
        self._client: genai.Client = genai.Client(vertexai=True, project=project, location=location)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one per input in input order.

        Args:
            texts: Texts to embed. May be empty.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in input order.

        Raises:
            StageError: If the Vertex API call fails.
        """
        if not texts:
            return []
        # gemini-embedding-* accepts only ONE input text per request on Vertex (a 400 fires
        # for any multi-text request), unlike the older text-embedding-* models which batch up
        # to ~250. Clamp the per-request size to 1 for that family so real (multi-chunk) corpora
        # don't fail; any other configured model keeps its batch size.
        per_request = 1 if self.model.startswith("gemini-embedding") else self.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), per_request):
            batch = texts[start : start + per_request]
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=self._types.EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=self.dim,
                    ),
                )
            except Exception as exc:  # normalize backend errors to a typed IndxError
                raise StageError("embed", str(exc)) from exc
            # Coerce at the adapter edge so no vendor float type leaks into core vectors,
            # then L2-normalize: gemini-embedding-001 only guarantees unit-norm at its
            # native 3072 width, so Matryoshka-truncated widths (e.g. 768) must be
            # re-normalized to keep cosine/dot scores comparable across backends
            # (matches e5.py / HashEmbedder, which always emit unit vectors).
            for embedding in result.embeddings:
                vec = [float(v) for v in embedding.values]
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0.0:
                    vec = [x / norm for x in vec]
                vectors.append(vec)
        return vectors
