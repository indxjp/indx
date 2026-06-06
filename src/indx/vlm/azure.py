"""AzureOpenAIVLM — Azure-hosted OpenAI vision adapter. Requires the `azure` extra (openai SDK).

Mirrors :class:`indx.vlm.gpt4o.GPT4oVLM` but targets an Azure OpenAI deployment via
``openai.AzureOpenAI`` (client construction mirrors :class:`indx.llm.azure.AzureOpenAILLM`).
Credentials and endpoint are read from kwargs or, when omitted, from the standard Azure
environment variables. The ``openai`` SDK is imported lazily so importing this module on a
clean install is always safe (coding-standards §2/§6.3).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import AzureOpenAI  # type: ignore[import-not-found]  # optional extra: azure

logger = logging.getLogger(__name__)

_SLOT = "vlm"
_NAME = "azure"
_EXTRA = "azure"

_DEFAULT_API_VERSION = "2024-10-21"
_DEFAULT_PROMPT = "Describe this image in detail."


class AzureOpenAIVLM:
    """Azure OpenAI multimodal chat adapter that describes images (satisfies the VLM protocol).

    Installed via ``indx[azure]``. Image bytes are encoded as a base64 ``data:`` URL and sent
    through the Chat Completions vision content shape against an Azure OpenAI deployment; only
    the extracted caption string is returned. The ``openai`` package is imported lazily inside
    the client factory so a clean ``pip install indx`` never pays for it (coding-standards §2),
    and the vendor response never escapes this adapter (§6.2). Secrets are read from the
    environment and are never logged or serialized (§8).
    """

    name = "azure"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
    ) -> None:
        """Configure the Azure OpenAI vision adapter.

        Args:
            api_key: Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY``.
            azure_endpoint: Resource endpoint URL. Defaults to ``AZURE_OPENAI_ENDPOINT``.
            api_version: Azure REST API version. Defaults to ``AZURE_OPENAI_API_VERSION``
                or a pinned recent version.
            deployment: GPT-4o(-class) vision deployment name. Defaults to
                ``AZURE_OPENAI_VLM_DEPLOYMENT``.

        Raises:
            MissingExtraError: If the ``azure`` extra (the ``openai`` SDK) is not installed.
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_VLM_DEPLOYMENT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or _DEFAULT_API_VERSION
        )
        self._client: AzureOpenAI | None = None

    def _ensure_client(self) -> AzureOpenAI:
        """Lazily build and cache the Azure OpenAI client.

        The API key is read from kwargs/the environment and handed to the SDK; it is never
        hard-coded or logged (coding-standards §8).

        Returns:
            The configured ``openai.AzureOpenAI`` client.

        Raises:
            StageError: If the API key or endpoint is missing.
        """
        if self._client is None:
            # optional extra: azure
            from openai import AzureOpenAI

            if not self._api_key:
                raise StageError(
                    "enrich",
                    "Azure OpenAI requires an API key (set AZURE_OPENAI_API_KEY or pass api_key=).",
                )
            if not self._azure_endpoint:
                raise StageError(
                    "enrich",
                    "Azure OpenAI requires an endpoint "
                    "(set AZURE_OPENAI_ENDPOINT or pass azure_endpoint=).",
                )
            logger.debug(
                "creating Azure OpenAI VLM client endpoint=%s api_version=%s deployment=%s",
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

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image using the Azure OpenAI vision deployment.

        Args:
            image: Raw image bytes (e.g. PNG/JPEG) to describe.
            prompt: Optional instruction guiding the description. Defaults to a generic
                "describe this image" prompt.

        Returns:
            The model's description text, or an empty string if the model returned none.

        Raises:
            StageError: If credentials/endpoint are missing or the deployment is unset.
        """
        if not self._deployment:
            raise StageError(
                "enrich",
                "Azure OpenAI VLM requires a deployment name "
                "(set AZURE_OPENAI_VLM_DEPLOYMENT or pass deployment=).",
            )
        client = self._ensure_client()
        b64 = base64.b64encode(image).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        resp = client.chat.completions.create(
            model=self._deployment,
            temperature=0.0,  # determinism (coding-standards §1.4)
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        # Convert vendor response -> core str AT THE EDGE. No openai type leaks upward.
        return resp.choices[0].message.content or ""
