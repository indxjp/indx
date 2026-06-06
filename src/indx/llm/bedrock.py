"""BedrockLLM — Amazon Bedrock LLM via the Converse API. Requires the `aws` extra (boto3).

Wraps ``bedrock-runtime``'s :meth:`converse` call behind the :class:`~indx.llm.base.LLM`
protocol. Per ``docs/cloud-providers-design.md`` §2/§5.1 the heavy ``boto3`` SDK is imported
lazily inside the client factory so a clean ``pip install indx`` never pays for it, and the
optional-extra gate (``require_extra``) runs at construction so selecting this backend without
``indx[aws]`` fails with an actionable :class:`~indx.errors.MissingExtraError`. Credentials come
from boto3's own chain (``AWS_*`` env / shared profile / SSO / role) — never read or logged here;
the adapter only supplies the non-secret ``region_name``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    import boto3  # type: ignore[import-not-found]  # optional extra: aws

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "bedrock"
_EXTRA = "aws"

# Bedrock model IDs drift fast and many require a region-prefixed inference profile
# (us./eu./apac.): in us-east-1 Claude Sonnet 4.x has no in-region endpoint, so the ``us.``
# geo profile is required (AWS Bedrock model card). Default to Claude Sonnet 4.6; the older
# 3.5 Sonnet v1 (20240620) reached end-of-life on Bedrock. Overridable via config ``model=``
# or the ``bedrock:<modelId>`` name suffix. NB: 4.x ids are suffix-less (no -YYYYMMDD-v1:0).
_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"


class BedrockLLM:
    """Amazon Bedrock Converse adapter (satisfies the ``LLM`` protocol structurally).

    Installed via ``indx[aws]`` (there is no separate ``indx[bedrock]`` — ``boto3`` is the one
    SDK for the whole AWS stack). The ``boto3`` package is imported lazily inside the client
    factory so a clean ``pip install indx`` never pays for it. Credentials are resolved by
    boto3's own auth chain and are never read or logged here (coding-standards §8).

    Attributes:
        name: Registry name for this backend (``"bedrock"``).
        model: The Bedrock ``modelId`` / inference-profile id in use.
    """

    name = "bedrock"

    def __init__(self, model: str | None = None, *, region: str | None = None) -> None:
        """Configure the Bedrock adapter, failing fast if the ``aws`` extra is absent.

        Args:
            model: Bedrock ``modelId`` or region-prefixed inference profile. Defaults to the
                ``us.``-prefixed Claude Sonnet 4.6 profile when ``None``.
            region: AWS region for the ``bedrock-runtime`` client. Defaults to the
                ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` environment variables (boto3 also
                resolves region from its own config when neither is set).
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "boto3")
        self.model = model or _DEFAULT_MODEL
        self._region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        # Any: holds the vendor bedrock-runtime client, which must never leak into core/.
        self._client: Any = None

    def _ensure_client(self) -> boto3.client:
        """Lazily build and cache the ``bedrock-runtime`` client.

        The ``boto3`` import is performed here (not at module top level) so importing this
        module is always safe on a light-core install.

        Returns:
            The cached ``boto3`` ``bedrock-runtime`` client.
        """
        if self._client is None:
            import boto3  # optional extra: aws

            logger.debug(
                "creating bedrock-runtime client region=%s model=%s",
                self._region,
                self.model,
            )
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` via the Bedrock Converse API.

        Args:
            prompt: The user prompt to complete.
            system: Optional system instruction. Bedrock takes ``system`` as a *list* of
                blocks, separate from ``messages`` — not an OpenAI-style system message.
            max_tokens: Maximum tokens to generate (sent as ``inferenceConfig.maxTokens``).
            temperature: Sampling temperature; defaults to ``0.0`` for determinism.

        Returns:
            The assistant message text, or an empty string if the model returned no text
            content block.

        Raises:
            StageError: If the Bedrock call fails. A ``ValidationException`` mentioning an
                inference profile is surfaced with a message telling the user to use the
                ``us.``-prefixed model id.
        """
        client = self._ensure_client()
        try:
            resp = client.converse(
                modelId=self.model,
                system=[{"text": system}] if system else [],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
        except Exception as exc:  # noqa: BLE001 - convert any vendor failure at the edge
            message = str(exc)
            if "ValidationException" in message and "inference profile" in message:
                raise StageError(
                    "enrich",
                    f"Bedrock rejected modelId {self.model!r}: this model requires a "
                    "region-prefixed inference profile. Use the us.-prefixed id "
                    "(e.g. us.anthropic.claude-sonnet-4-6).",
                ) from exc
            raise StageError("enrich", message) from exc
        # Convert the vendor response -> core ``str`` AT THE EDGE: no boto3 dict escapes into
        # core/. Iterate content blocks defensively (first with a "text" key) so tool-use
        # responses don't KeyError.
        content = resp["output"]["message"]["content"]
        for block in content:
            if "text" in block:
                return str(block["text"])
        return ""
