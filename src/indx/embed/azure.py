"""AzureOpenAIEmbedder — Azure-hosted OpenAI embeddings. Requires the `azure` extra (openai SDK).

Mirrors :class:`indx.embed.openai.OpenAIEmbedder` but targets an Azure OpenAI deployment via
``openai.AzureOpenAI`` (the same client construction as :class:`indx.llm.azure.AzureOpenAILLM`).
Credentials and endpoint are read from kwargs or, when omitted, from the standard Azure environment
variables. The ``openai`` SDK is imported lazily so importing this module on a clean install is
always safe (coding-standards §2/§6.3).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from indx.embed._batch import MAX_TOKENS_PER_REQUEST, embed_in_batches
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import AzureOpenAI  # type: ignore[import-not-found]  # optional extra: azure

logger = logging.getLogger(__name__)

_SLOT = "embedder"
_NAME = "azure"
_EXTRA = "azure"

_DEFAULT_API_VERSION = "2024-10-21"
_DEFAULT_DIM = 1536

# Output dimensions inferred from substrings of the Azure *deployment* name. Deployment names are
# user-chosen, so this is best-effort; callers can override with ``dimensions=`` when it differs.
_DIM_BY_SUBSTRING: tuple[tuple[str, int], ...] = (
    ("3-large", 3072),
    ("3-small", 1536),
    ("ada", 1536),
)


def _infer_dim(deployment: str) -> int:
    """Infer the embedding width from the deployment name, defaulting to 1536."""
    lowered = deployment.lower()
    for needle, dim in _DIM_BY_SUBSTRING:
        if needle in lowered:
            return dim
    return _DEFAULT_DIM


class AzureOpenAIEmbedder:
    """Embedder backed by an Azure OpenAI embeddings deployment (the ``Embedder`` protocol).

    Installed via ``indx[azure]``. The ``openai`` package is imported lazily inside the client
    factory so a clean ``pip install indx`` never pays for it (coding-standards §2). Secrets are
    read from the environment and are never logged or serialized (§8).

    Attributes:
        name: The Azure deployment name, recorded into the archive manifest.
        dim: Output dimensionality of the deployment (inferred or overridden via ``dimensions=``).
    """

    name = "azure"
    dim = _DEFAULT_DIM

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 256,
        max_tokens_per_request: int = MAX_TOKENS_PER_REQUEST,
    ) -> None:
        """Configure the Azure OpenAI embedder.

        Args:
            model: Embedding deployment name to invoke. Falls back to ``deployment`` and then to the
                ``AZURE_OPENAI_EMBED_DEPLOYMENT`` environment variable.
            api_key: Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY``.
            azure_endpoint: Resource endpoint URL. Defaults to ``AZURE_OPENAI_ENDPOINT``.
            api_version: Azure REST API version. Defaults to ``AZURE_OPENAI_API_VERSION`` or a
                pinned recent version.
            deployment: Alias for ``model`` (the Azure deployment name).
            dimensions: Optional output-dimension override (3-small/3-large support truncation when
                ``api_version`` is at least ``2024-10-21``). When omitted, ``dim`` is inferred from
                the deployment name.
            batch_size: Maximum number of texts sent per ``embeddings.create`` request.
            max_tokens_per_request: Token ceiling per request. Batches are sub-split to stay under
                it so dense (e.g. CJK) text does not overflow Azure OpenAI's 300k-token cap.

        Raises:
            MissingExtraError: If the ``azure`` extra is not installed.
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        self._deployment = model or deployment or os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or _DEFAULT_API_VERSION
        )
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_tokens_per_request = max_tokens_per_request
        # Record the *actual* deployment in the manifest; reflect its real width.
        self.name = self._deployment or _NAME
        inferred = _infer_dim(self._deployment) if self._deployment else _DEFAULT_DIM
        self.dim = dimensions or inferred
        self._client: AzureOpenAI | None = None

    def _ensure_client(self) -> AzureOpenAI:
        """Lazily build and cache the Azure OpenAI client.

        Returns:
            The configured ``openai.AzureOpenAI`` client.

        Raises:
            StageError: If the deployment, API key, or endpoint is missing.
        """
        if self._client is None:
            # optional extra: azure
            from openai import AzureOpenAI

            if not self._deployment:
                raise StageError(
                    "embed",
                    "Azure OpenAI embeddings requires a deployment name "
                    "(set AZURE_OPENAI_EMBED_DEPLOYMENT or pass model=/deployment=).",
                )
            if not self._api_key:
                raise StageError(
                    "embed",
                    "Azure OpenAI requires an API key (set AZURE_OPENAI_API_KEY or pass api_key=).",
                )
            if not self._azure_endpoint:
                raise StageError(
                    "embed",
                    "Azure OpenAI requires an endpoint "
                    "(set AZURE_OPENAI_ENDPOINT or pass azure_endpoint=).",
                )
            logger.debug(
                "creating Azure OpenAI client endpoint=%s api_version=%s deployment=%s",
                self._azure_endpoint,
                self._api_version,
                self._deployment,
            )
            self._client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors, one per input in input order.

        Args:
            texts: Texts to embed. May be empty.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in input order.

        Raises:
            StageError: If credentials/endpoint/deployment are missing or the API call fails.
        """
        if not texts:
            return []
        client = self._ensure_client()

        def call(batch: list[str]) -> list[list[float]]:
            kwargs: dict[str, object] = {"model": self._deployment, "input": batch}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = client.embeddings.create(**kwargs)
            # The API echoes each item's input index; sort to guarantee input order, then convert
            # the vendor response → core vectors AT THE EDGE. No openai/float type leaks upward.
            return [
                [float(v) for v in item.embedding]
                for item in sorted(response.data, key=lambda d: d.index)
            ]

        try:
            return embed_in_batches(
                texts,
                max_items=self._batch_size,
                max_tokens=self._max_tokens_per_request,
                call=call,
            )
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("embed", str(exc)) from exc
