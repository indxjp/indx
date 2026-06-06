"""LiteLLMEmbedder — reach any embedding provider through one adapter (``litellm:<model>``).

The embedding counterpart to :class:`~indx.llm.litellm.LiteLLMClient`. LiteLLM's ``embedding``
call normalizes every provider, so one indx embedder covers on-prem (Ollama, vLLM) and the
managed clouds (Bedrock Titan, Azure OpenAI, Vertex, Cohere, OpenAI, …). Select the provider
with LiteLLM's ``provider/model`` string via the registry's ``:model`` suffix::

    indx ./docs --out ./out --embedder litellm:bedrock/amazon.titan-embed-text-v2:0
    indx ./docs --out ./out --embedder litellm:azure/my-embedding-deployment
    indx ./docs --out ./out --embedder litellm:ollama/nomic-embed-text      # on-prem

The output dimension is provider-specific and usually unknown up front, so it is **discovered
from the first response** and recorded into the manifest. Pass ``dim=`` explicitly to pin it
when you already know it. Credentials come from each provider's standard environment; the
heavy ``litellm`` SDK is imported lazily and gated by :func:`~indx.utils.lazy.require_extra`.
"""

from __future__ import annotations

import os
from typing import Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

_DEFAULT_MODEL = "text-embedding-3-small"
# Fallback width recorded before the first embed call resolves the real dimension.
_DEFAULT_DIM = 1536


class LiteLLMEmbedder:
    """Embedder backed by LiteLLM's unified ``embedding`` API.

    Installed via ``indx[litellm]``. ``name`` records the actual model into the archive
    manifest; ``dim`` reflects the model's true width once the first batch is embedded.

    Attributes:
        name: Stable model identifier recorded into the manifest.
        dim: Output dimensionality (discovered from the first response unless pinned).
    """

    name = "litellm"
    dim = _DEFAULT_DIM

    def __init__(
        self,
        model: str | None = None,
        *,
        dim: int | None = None,
        api_base: str | None = None,
        batch_size: int = 256,
    ) -> None:
        """Construct the embedder, failing fast if the ``litellm`` extra is absent.

        Args:
            model: LiteLLM ``provider/model`` string. Defaults to ``text-embedding-3-small``.
            dim: Pin the output dimension. When ``None``, it is discovered from the first batch.
            api_base: Optional base URL for self-hosted/on-prem endpoints. Falls back to the
                ``LITELLM_API_BASE`` environment variable.
            batch_size: Number of texts sent per ``embedding`` request.
        """
        require_extra("embedder", "litellm", "litellm", "litellm")
        self.model = model or _DEFAULT_MODEL
        self.name = self.model  # record the *actual* model in the manifest
        self.dim = dim or _DEFAULT_DIM
        self._dim_pinned = dim is not None
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one per input in input order.

        Raises:
            StageError: If the LiteLLM embedding call fails.
        """
        if not texts:
            return []
        import litellm  # type: ignore[import-not-found]  # optional extra: litellm

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            try:
                response = litellm.embedding(**kwargs)
            except Exception as exc:  # normalize backend errors to a typed IndxError
                raise StageError("embed", str(exc)) from exc
            for item in response.data:
                # LiteLLM returns dict-shaped items ({"embedding": [...], "index": i}); some
                # providers return attribute objects. Coerce at the edge so no vendor type leaks.
                embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                vectors.append([float(v) for v in embedding])

        # Record the true width discovered from the response unless the caller pinned it.
        if vectors and not self._dim_pinned:
            self.dim = len(vectors[0])
        return vectors
