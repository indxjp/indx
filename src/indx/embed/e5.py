"""E5Embedder — local sentence-transformers embedder. Requires the ``e5`` extra.

Loads an E5 model (default ``intfloat/e5-base-v2``, dense dimension 768) through
**sentence-transformers**, which drags in **torch** — so the heavy import lives behind the
``indx[e5]`` extra and is *never* imported at module top level (coding-standards §2). Output is
L2-normalized to match :class:`~indx.embed.hash_embedder.HashEmbedder`, so cosine/dot scores are
comparable across backends (07-embeddings §"L2 normalization consistency").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from sentence_transformers import (  # type: ignore[import-not-found]  # optional extra: e5
        SentenceTransformer,
    )


class E5Embedder:
    """Local E5 embedder backed by sentence-transformers.

    Satisfies the :class:`~indx.embed.base.Embedder` Protocol: exposes ``name``/``dim`` and a
    batch-shaped ``embed``. The vendor model is loaded lazily in ``__init__`` so importing this
    module is always safe; selecting the slot without the ``e5`` extra raises
    :class:`~indx.errors.MissingExtraError` at construction with the exact install hint.

    Attributes:
        name: The actual model id (e.g. ``"intfloat/e5-base-v2"``), recorded into the archive
            manifest so the ``.indx`` artifact is self-describing.
        dim: The resolved output dimensionality, read from the model rather than hard-coded so a
            different model can't silently desync the manifest.
    """

    name = "e5"
    dim = 768  # intfloat/e5-base-v2 dense dimension; refined from the model in __init__

    def __init__(
        self,
        model_id: str = "intfloat/e5-base-v2",
        *,
        batch_size: int = 12,
        device: str | None = None,
    ) -> None:
        """Load the E5 model, failing loud if the ``e5`` extra is absent.

        Args:
            model_id: Hugging Face model id to load. Recorded as ``self.name``.
            batch_size: Batch size handed to the backend's ``encode`` so it sub-batches a whole
                list in one CPU/GPU pass (coding-standards §14).
            device: Optional device string (e.g. ``"cuda:0"``); ``None`` lets the backend choose.

        Raises:
            MissingExtraError: If the ``e5`` extra (sentence-transformers + torch) is not installed.
        """
        # Fail with one actionable message if the extra is absent; keep this first so importing
        # the module stays safe (coding-standards §6.3 import-safety).
        require_extra("embedder", "e5", "e5", "sentence_transformers")
        from sentence_transformers import SentenceTransformer  # optional extra: e5

        self.name = model_id  # record the *actual* model in the manifest, not a generic label
        self.batch_size = batch_size
        self._model: SentenceTransformer = SentenceTransformer(model_id, device=device)
        # sentence-transformers v5.4 renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension; prefer the new name, fall back for older installs.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        resolved = get_dim()
        if resolved is not None:
            self.dim = int(resolved)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into one L2-normalized vector each, in input order.

        Args:
            texts: The batch of texts to embed. May be empty.

        Returns:
            One ``list[float]`` per input text, each of length ``self.dim``. Values are plain
            Python floats (vendor ndarray rows are converted at the adapter edge).

        Raises:
            StageError: If the backend's ``encode`` call fails.
        """
        if not texts:
            return []
        try:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # cosine-ready, matches HashEmbedder's L2 norm
                convert_to_numpy=True,
            )
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("embed", str(exc)) from exc
        return [[float(v) for v in row] for row in vecs]
