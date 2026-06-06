"""BedrockEmbedder — Amazon Bedrock embedding models. Requires the ``aws`` extra.

Wraps the Bedrock Runtime ``invoke_model`` API behind the tiny
:class:`~indx.embed.base.Embedder` Protocol (``name``, ``dim``, ``embed``). The heavy
``boto3`` SDK is imported lazily inside ``__init__`` so importing this module is always
safe; selecting the slot without the extra installed raises
:class:`~indx.errors.MissingExtraError` with an actionable ``pip install indx[aws]`` hint
(file-architecture §5, coding-standards §6).

Credentials come from boto3's own credential chain (env / shared config / instance role);
this module never reads or logs keys. Vendor request/response payloads (JSON blobs and the
``bedrock-runtime`` client) live only inside this module: they are converted to plain
``list[list[float]]`` core vectors here (coding-standards §6.2).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

logger = logging.getLogger(__name__)

# Output dimensions per Bedrock embedding model. Titan v2 is variable-width (see the
# ``dimensions`` knob below); these are the family defaults / Cohere-on-Bedrock widths.
_MODEL_DIMS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}

_DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
# Dimensions the Titan v2 ``dimensions`` knob accepts.
_TITAN_V2_DIMS = frozenset({256, 512, 1024})


class BedrockEmbedder:
    """Embedder backed by Amazon Bedrock's ``invoke_model`` embedding API.

    Uses boto3's native credential chain (never hard-coded, never logged). The model id is
    recorded as ``name`` so the archive manifest is self-describing, and ``dim`` reflects
    the model's actual output width. Titan v2 honors a ``dimensions`` knob (1024 by
    default, also 512/256); ``dim`` mirrors whatever was selected.

    Attributes:
        name: The Bedrock model id used (e.g. ``"amazon.titan-embed-text-v2:0"``).
        dim: The output vector dimensionality for the selected model.
    """

    name: str
    dim: int

    def __init__(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        input_type: str = "search_document",
        region: str | None = None,
    ) -> None:
        """Construct the embedder and the underlying ``bedrock-runtime`` client.

        Args:
            model: Bedrock embedding model id. Defaults to
                ``"amazon.titan-embed-text-v2:0"`` (a bare id — embedders do not use
                inference profiles).
            dimensions: Titan v2 output width — one of 256, 512, or 1024. When ``None``
                (the default) Titan v2 uses 1024. Ignored for non-Titan-v2 models, whose
                width is fixed.
            input_type: Cohere-on-Bedrock ``input_type`` hint. Documents being indexed use
                ``"search_document"`` (the default); query embedding callers may pass
                ``"search_query"``. Ignored by Titan models.
            region: AWS region for the ``bedrock-runtime`` client. When ``None``, boto3
                resolves the region from its own config chain.

        Raises:
            MissingExtraError: If the ``aws`` extra (``boto3``) is not installed.
            StageError: If ``dimensions`` is not a value Titan v2 accepts.
        """
        # First statement: fail with one actionable message if the extra is absent,
        # so importing this module stays safe (coding-standards §6.3).
        require_extra("embedder", "bedrock", "aws", "boto3")
        import boto3  # type: ignore[import-not-found]  # optional extra: aws (lazy)

        resolved_model = model or _DEFAULT_MODEL
        self.name = resolved_model
        self.model = resolved_model
        self.input_type = input_type

        if resolved_model == "amazon.titan-embed-text-v2:0":
            if dimensions is None:
                self.dim = 1024
            elif dimensions in _TITAN_V2_DIMS:
                self.dim = dimensions
            else:
                raise StageError(
                    "embed",
                    f"Titan v2 supports dimensions 256/512/1024, not {dimensions}.",
                )
        else:
            self.dim = _MODEL_DIMS.get(resolved_model, 1024)

        # Resolve the region from the env when not passed, so a default client isn't created
        # region-less (boto3 then raises NoRegionError on first call) — mirrors BedrockLLM.
        resolved_region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        # Never log credentials — only the (non-secret) model shape.
        logger.debug("initializing Bedrock embedder model=%s dim=%d", self.name, self.dim)
        self._client: Any = boto3.client("bedrock-runtime", region_name=resolved_region)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into one vector per input, in input order.

        Titan models embed one ``inputText`` per call (no batch), so this loops; the
        Cohere-on-Bedrock models accept the whole batch in one call.

        Args:
            texts: The batch of texts to embed.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in the same order as
            ``texts``. An empty input yields an empty list.

        Raises:
            StageError: If the Bedrock API call fails or returns a malformed response.
        """
        if not texts:
            return []
        if self.model.startswith("cohere."):
            return self._embed_cohere(texts)
        return self._embed_titan(texts)

    def _embed_titan(self, texts: list[str]) -> list[list[float]]:
        """Embed via Titan: one ``inputText`` per ``invoke_model`` call."""
        vectors: list[list[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
            try:
                response = self._client.invoke_model(modelId=self.model, body=body)
                out = json.loads(response["body"].read())
            except Exception as exc:  # normalize vendor errors to a typed IndxError
                raise StageError("embed", f"Bedrock invoke_model failed: {exc}") from exc
            embedding = out.get("embedding")
            if embedding is None:
                raise StageError("embed", "Bedrock response missing 'embedding'.")
            vectors.append([float(x) for x in embedding])
        return vectors

    def _embed_cohere(self, texts: list[str]) -> list[list[float]]:
        """Embed via Cohere-on-Bedrock: the whole batch in one ``invoke_model`` call."""
        body = json.dumps({"texts": texts, "input_type": self.input_type})
        try:
            response = self._client.invoke_model(modelId=self.model, body=body)
            out = json.loads(response["body"].read())
        except Exception as exc:  # normalize vendor errors to a typed IndxError
            raise StageError("embed", f"Bedrock invoke_model failed: {exc}") from exc
        emb = out.get("embeddings")
        if emb is None:
            raise StageError("embed", "Bedrock response missing 'embeddings'.")
        # Cohere v3 returns {"float": [...]} ; older shapes return a bare list.
        vectors = emb["float"] if isinstance(emb, dict) else emb
        return [[float(x) for x in vector] for vector in vectors]
