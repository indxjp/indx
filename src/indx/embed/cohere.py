"""CohereEmbedder — Cohere embedding API. Requires the ``cohere`` extra.

Wraps :class:`cohere.Client` behind the tiny :class:`~indx.embed.base.Embedder`
Protocol (``name``, ``dim``, ``embed``). The heavy vendor SDK is imported lazily
inside ``__init__`` so importing this module is always safe; selecting the slot
without the extra installed raises :class:`~indx.errors.MissingExtraError` with an
actionable ``pip install indx[cohere]`` hint (file-architecture §5, coding-standards §6).

Vendor types never leak past this module: the Cohere client and its response objects
are converted to plain ``list[list[float]]`` core vectors here (coding-standards §6.2).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    import cohere  # type: ignore[import-not-found, unused-ignore]  # optional extra: cohere

logger = logging.getLogger(__name__)

# Output dimensions per Cohere embedding model family (embed v3 = 1024).
_MODEL_DIMS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

_DEFAULT_MODEL = "embed-english-v3.0"


class CohereEmbedder:
    """Embedder backed by Cohere's hosted ``embed`` API.

    Reads the API key from the ``COHERE_API_KEY`` environment variable (never
    hard-coded, never logged). The model id is recorded as ``name`` so the archive
    manifest is self-describing, and ``dim`` reflects the model's actual output width.

    Attributes:
        name: The Cohere model id used (e.g. ``"embed-english-v3.0"``).
        dim: The output vector dimensionality for the selected model.
    """

    name: str
    dim: int

    def __init__(
        self,
        model: str | None = None,
        *,
        input_type: str = "search_document",
        api_key: str | None = None,
    ) -> None:
        """Construct the embedder and the underlying Cohere client.

        Args:
            model: Cohere embedding model id. Defaults to ``"embed-english-v3.0"``.
            input_type: Cohere ``input_type`` hint for v3 models. Documents being
                indexed use ``"search_document"`` (the default); query embedding
                callers may pass ``"search_query"``.
            api_key: Explicit API key. When ``None`` (the default), the key is read
                from the ``COHERE_API_KEY`` environment variable.

        Raises:
            MissingExtraError: If the ``cohere`` extra is not installed.
            StageError: If no API key can be resolved from the argument or the
                ``COHERE_API_KEY`` environment variable.
        """
        # First statement: fail with one actionable message if the extra is absent,
        # so importing this module stays safe (coding-standards §6.3).
        require_extra("embedder", "cohere", "cohere", "cohere")
        import cohere  # optional extra: cohere (lazy; see TYPE_CHECKING import above)

        resolved_model = model or _DEFAULT_MODEL
        self.name = resolved_model
        self.dim = _MODEL_DIMS.get(resolved_model, 1024)
        self.input_type = input_type

        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise StageError(
                "embed",
                "Cohere requires an API key: set the COHERE_API_KEY environment variable.",
            )
        # Never log the key or its value — only the (non-secret) model shape.
        logger.debug("initializing Cohere embedder model=%s dim=%d", self.name, self.dim)
        self._client: cohere.Client = cohere.Client(key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into one vector per input, in input order.

        Args:
            texts: The batch of texts to embed.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in the same
            order as ``texts``. An empty input yields an empty list.

        Raises:
            StageError: If the Cohere API call fails or returns a malformed response.
        """
        if not texts:
            return []
        try:
            response = self._client.embed(
                texts=texts,
                model=self.name,
                input_type=self.input_type,
            )
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("embed", f"Cohere embed call failed: {exc}") from exc

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise StageError("embed", "Cohere response missing 'embeddings'.")
        return [[float(value) for value in vector] for vector in embeddings]
