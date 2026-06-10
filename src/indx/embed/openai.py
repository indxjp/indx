"""OpenAIEmbedder — default embedder (``openai:text-embedding-3-small``).

Wraps the OpenAI Embeddings API behind the tiny :class:`~indx.embed.base.Embedder`
Protocol. The ``openai`` SDK is heavy/optional, so it is imported lazily inside
``__init__`` and gated by :func:`~indx.utils.lazy.require_extra`; importing this module
is always safe (file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from indx.embed._batch import MAX_TOKENS_PER_REQUEST, embed_in_batches
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import OpenAI  # type: ignore[import-not-found]  # optional extra: openai

# Output dimensions per known OpenAI embedding model.
_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """Embedder backed by the OpenAI Embeddings API.

    The default model is ``text-embedding-3-small`` (1536-dim). The API key is read
    from the ``OPENAI_API_KEY`` environment variable by the SDK; it is never logged or
    stored on this object.

    Attributes:
        name: Stable model identifier recorded into the archive manifest.
        dim: Output dimensionality of the selected model.
    """

    name = "openai"
    dim = 1536

    def __init__(
        self,
        model: str | None = None,
        *,
        batch_size: int = 256,
        max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST,
    ) -> None:
        """Construct the embedder and its OpenAI client.

        Args:
            model: OpenAI embedding model id. Defaults to ``text-embedding-3-small``.
            batch_size: Maximum number of texts sent per ``embeddings.create`` request.
            max_tokens_per_request: Token ceiling per request. Batches are sub-split to stay
                under it so dense (e.g. CJK) text does not overflow OpenAI's 300k-token cap.

        Raises:
            MissingExtraError: If the ``openai`` extra is not installed.
        """
        # Fail with one actionable message if the extra is absent. Kept as the first
        # statement so merely importing this module stays safe (coding-standards §6.3).
        require_extra("embedder", "openai", "openai", "openai")
        from openai import OpenAI  # optional extra: openai (lazy; see TYPE_CHECKING import)

        self.model = model or _DEFAULT_MODEL
        self.name = self.model  # record the *actual* model in the manifest
        self.dim = _MODEL_DIMS.get(self.model, 1536)  # reflect the model's actual width
        self.batch_size = batch_size
        self.max_tokens_per_request = max_tokens_per_request
        # The SDK reads OPENAI_API_KEY from the environment; the key is never stored here.
        self._client: OpenAI = OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one per input in input order.

        Args:
            texts: Texts to embed. May be empty.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in input order.

        Raises:
            StageError: If the OpenAI API call fails.
        """
        if not texts:
            return []

        def call(batch: list[str]) -> list[list[float]]:
            response = self._client.embeddings.create(input=batch, model=self.model)
            # The API echoes each item's input index; sort to guarantee input order, then
            # coerce at the adapter edge so no vendor float type leaks into core vectors.
            return [
                [float(v) for v in item.embedding]
                for item in sorted(response.data, key=lambda d: d.index)
            ]

        try:
            return embed_in_batches(
                texts,
                max_items=self.batch_size,
                max_tokens=self.max_tokens_per_request,
                call=call,
            )
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("embed", str(exc)) from exc
