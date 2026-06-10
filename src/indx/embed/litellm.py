"""LiteLLMEmbedder — reach any embedding provider through one adapter (``litellm:<model>``).

The embedding counterpart to :class:`~indx.llm.litellm.LiteLLMClient`. LiteLLM's ``embedding``
call normalizes every provider, so one indx embedder covers on-prem (Ollama, vLLM) and the
managed clouds (Bedrock Titan, Azure OpenAI, Vertex, Cohere, OpenAI, …). Select the provider
with LiteLLM's ``provider/model`` string via the registry's ``:model`` suffix::

    indx ./docs --out ./out --embedder litellm:bedrock/amazon.titan-embed-text-v2:0
    indx ./docs --out ./out --embedder litellm:azure/my-embedding-deployment
    indx ./docs --out ./out --embedder litellm:ollama/nomic-embed-text      # on-prem

The output dimension is provider-specific and usually unknown up front, so it is **discovered
from a single-token probe the first time** :attr:`~LiteLLMEmbedder.dim` **is read** (and from
the first real batch otherwise) and recorded into the manifest. The probe matters because the
pipeline reads ``embedder.dim`` to pre-size dimension-locked stores (qdrant/pgvector/…) before
any document is embedded; a stale placeholder would size the store at the wrong width. Pass
``dim=`` explicitly to pin it when you already know it. Credentials come from each provider's
standard environment; the heavy ``litellm`` SDK is imported lazily and gated by
:func:`~indx.utils.lazy.require_extra`.
"""

from __future__ import annotations

import os
from typing import Any

from indx.embed._batch import MAX_TOKENS_PER_REQUEST, embed_in_batches
from indx.errors import StageError
from indx.utils.lazy import require_extra

_DEFAULT_MODEL = "text-embedding-3-small"
# Fallback width recorded before the first embed call resolves the real dimension.
_DEFAULT_DIM = 1536


class LiteLLMEmbedder:
    """Embedder backed by LiteLLM's unified ``embedding`` API.

    Installed via ``indx[litellm]``. ``name`` records the actual model into the archive
    manifest; ``dim`` reflects the model's true width, resolved on first read via a single-token
    probe so the pipeline can pre-size dimension-locked stores correctly.

    Attributes:
        name: Stable model identifier recorded into the manifest.
        dim: Output dimensionality (probed on first read from the provider unless pinned).
    """

    name = "litellm"

    def __init__(
        self,
        model: str | None = None,
        *,
        dim: int | None = None,
        api_base: str | None = None,
        batch_size: int = 256,
        max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST,
    ) -> None:
        """Construct the embedder, failing fast if the ``litellm`` extra is absent.

        Args:
            model: LiteLLM ``provider/model`` string. Defaults to ``text-embedding-3-small``.
            dim: Pin the output dimension. When ``None``, it is discovered from the first batch.
            api_base: Optional base URL for self-hosted/on-prem endpoints. Falls back to the
                ``LITELLM_API_BASE`` environment variable.
            batch_size: Maximum number of texts sent per ``embedding`` request.
            max_tokens_per_request: Token ceiling per request. Batches are sub-split to stay
                under it so dense (e.g. CJK) text does not overflow the provider's token cap.
        """
        require_extra("embedder", "litellm", "litellm", "litellm")
        self.model = model or _DEFAULT_MODEL
        self.name = self.model  # record the *actual* model in the manifest
        # ``dim`` is resolved lazily (see the ``dim`` property): pinned widths take effect
        # immediately, otherwise the true width is discovered from a probe on first read.
        self._dim: int | None = dim
        self._dim_pinned = dim is not None
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")
        self.batch_size = batch_size
        self.max_tokens_per_request = max_tokens_per_request

    @property
    def dim(self) -> int:
        """Output width, probed from the provider on first read when not pinned.

        The pipeline reads this to pre-size dimension-locked stores (qdrant/pgvector/…) before
        any document is embedded, so an unresolved placeholder would create a wrong-width store
        that later rejects the real vectors. A single-token probe resolves and caches the true
        width on the first read (a failed probe surfaces as :class:`~indx.errors.StageError`);
        a provider returning no probe data falls back to the placeholder without caching so a
        later read can retry. Pinning ``dim=`` skips the probe entirely.
        """
        if self._dim is not None:
            return self._dim
        probe = self.embed(["probe"])
        if probe:
            return self.dim  # embed() cached the discovered width on self._dim
        return _DEFAULT_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one per input in input order.

        Raises:
            StageError: If the LiteLLM embedding call fails.
        """
        if not texts:
            return []
        import litellm  # type: ignore[import-not-found]  # optional extra: litellm

        def call(batch: list[str]) -> list[list[float]]:
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = litellm.embedding(**kwargs)

            # LiteLLM returns dict-shaped items ({"embedding": [...], "index": i}); some
            # providers return attribute objects. Sort by the per-item ``index`` to guarantee
            # input order (LiteLLM aggregates providers whose ``data`` order is not positional),
            # then coerce at the edge so no vendor type leaks.
            def _index(item: Any) -> int:
                return int(item["index"] if isinstance(item, dict) else item.index)

            vecs: list[list[float]] = []
            for item in sorted(response.data, key=_index):
                embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                vecs.append([float(v) for v in embedding])
            return vecs

        try:
            vectors = embed_in_batches(
                texts,
                max_items=self.batch_size,
                max_tokens=self.max_tokens_per_request,
                call=call,
            )
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("embed", str(exc)) from exc

        # Record the true width discovered from the response unless the caller pinned it.
        if vectors and not self._dim_pinned:
            self._dim = len(vectors[0])
        return vectors
