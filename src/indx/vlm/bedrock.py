"""BedrockVLM — Amazon Bedrock vision adapter (Converse API). Requires the `aws` extra.

The ``boto3`` SDK is imported lazily inside :meth:`BedrockVLM._ensure_client` so a clean
``pip install indx`` never pays for it (coding-standards §2). Construction calls
``require_extra`` first, so selecting this backend without the extra fails fast with the
friendly :class:`~indx.errors.MissingExtraError` (file-architecture §5).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

_SLOT = "vlm"
_NAME = "bedrock"
_EXTRA = "aws"

# Default to a region-prefixed inference profile: many Bedrock model ids require one — in
# us-east-1 Claude Sonnet 4.x has no in-region endpoint, so the ``us.`` geo profile is
# required (AWS Bedrock model card, cloud-providers-design §5.1). Vision-capable Claude
# Sonnet 4.6; the older 3.5 Sonnet v1 (20240620) reached end-of-life on Bedrock. Shared
# with BedrockLLM; override via config. NB: 4.x ids are suffix-less (no -YYYYMMDD-v1:0).
_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
_DEFAULT_PROMPT = "Describe this image in detail."
# Bedrock Converse image formats; png is the safe default (no sniffing).
_FORMATS = frozenset({"png", "jpeg", "gif", "webp"})

logger = logging.getLogger(__name__)


class BedrockVLM:
    """Amazon Bedrock Converse adapter that describes images (satisfies the VLM protocol).

    Installed via ``indx[aws]``. Raw image bytes are sent through the Converse image content
    block — boto3 base64-encodes the payload for us, so (unlike the OpenAI data-URL path in
    ``gpt4o.py``) we pass the bytes verbatim. Only the extracted caption string is returned;
    the vendor response dict never escapes this adapter (coding-standards §6.2).
    """

    name = "bedrock"

    def __init__(
        self,
        model: str | None = None,
        *,
        region: str | None = None,
        profile: str | None = None,
        image_format: str = "png",
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The Bedrock vision-capable model id (often a region-prefixed inference
                profile). Defaults to a Claude 3.5 Sonnet profile.
            region: AWS region for the ``bedrock-runtime`` client. Defaults to the boto3
                chain (``AWS_REGION``/``AWS_DEFAULT_REGION``).
            profile: Optional named AWS profile passed to ``boto3.Session``.
            image_format: Bedrock image format, one of ``png``/``jpeg``/``gif``/``webp``.

        Raises:
            MissingExtraError: If the ``aws`` extra (``boto3``) is not installed.
            StageError: If ``image_format`` is not a supported Bedrock image format.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "boto3")
        if image_format not in _FORMATS:
            raise StageError(
                "enrich",
                f"unsupported image format {image_format!r}; use one of {sorted(_FORMATS)}",
            )
        self.model = model or _DEFAULT_MODEL
        # Mirror BedrockLLM: resolve the region from the env when not passed, so a default
        # client isn't created region-less (boto3 then raises NoRegionError on first call).
        self._region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        self._profile = profile
        self._format = image_format
        self._client: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_client(self) -> Any:  # Any: vendor client, never leaks into core/
        """Lazily build and cache the ``bedrock-runtime`` client.

        Credentials are discovered by boto3 itself via the standard chain (env / shared
        config / SSO / role); the adapter only passes the non-secret ``region``/``profile``
        and never reads or logs keys (coding-standards §8).

        Returns:
            The cached ``bedrock-runtime`` boto3 client.
        """
        if self._client is None:
            import boto3  # type: ignore[import-not-found]  # optional extra: aws (lazy)

            session = boto3.Session(profile_name=self._profile)
            self._client = session.client("bedrock-runtime", region_name=self._region)
            logger.debug("vlm backend=%s model=%s region=%s", _NAME, self.model, self._region)
        return self._client

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image using the Bedrock Converse vision model.

        Args:
            image: Raw image bytes (e.g. PNG/JPEG) to describe. Passed verbatim — boto3
                base64-encodes the payload, so we must NOT pre-encode it.
            prompt: Optional instruction guiding the description. Defaults to a generic
                "describe this image" prompt.

        Returns:
            The model's description text, or an empty string if the model returned none.

        Raises:
            StageError: If the Bedrock call fails (wrapped at the edge).
        """
        client = self._ensure_client()
        try:
            resp = client.converse(
                modelId=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt or _DEFAULT_PROMPT},
                            {"image": {"format": self._format, "source": {"bytes": image}}},
                        ],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 — convert any vendor failure at the edge
            raise StageError("enrich", f"Bedrock converse (vlm) failed: {exc}") from exc
        # Convert vendor response -> core str AT THE EDGE. Iterate content blocks
        # defensively so tool-use / non-text blocks don't raise.
        content = resp.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "text" in block:
                return str(block["text"])
        return ""
