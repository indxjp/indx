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
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra
from indx.vlm.gpt4o import _sniff_mime

if TYPE_CHECKING:
    from openai import AzureOpenAI  # type: ignore[import-not-found]  # optional extra: azure

logger = logging.getLogger(__name__)

_SLOT = "vlm"
_NAME = "azure"
_EXTRA = "azure"

_DEFAULT_API_VERSION = "2024-10-21"
_DEFAULT_PROMPT = "Describe this image in detail."


def _unsupported_param(exc: Exception) -> str | None:
    """Name the request parameter an OpenAI/Azure 400 says is unsupported, or ``None``.

    Matches the SDK's ``BadRequestError`` (status 400) whose message flags ``temperature`` or
    ``reasoning_effort`` as unsupported, without importing the optional ``openai`` types at
    module top (the SDK is an extra). Used to drive the retry in
    :meth:`AzureOpenAIVLM._create_resilient` (mirrors :func:`indx.llm.azure._unsupported_param`).
    """
    if type(exc).__name__ != "BadRequestError" and getattr(exc, "status_code", None) != 400:
        return None
    msg = str(getattr(exc, "message", None) or exc).lower()
    signals = ("support", "unsupported", "unrecognized", "unknown parameter", "not supported")
    if not any(token in msg for token in signals):
        return None
    if "temperature" in msg:
        return "temperature"
    if "reasoning_effort" in msg:
        return "reasoning_effort"
    return None


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

    @staticmethod
    def _is_reasoning_model(deployment: str) -> bool:
        """Return ``True`` if the deployment name denotes a gpt-5/o-series reasoning model.

        These reject custom ``temperature`` on Chat Completions (mirrors
        :meth:`indx.llm.azure.AzureOpenAILLM._is_reasoning_model`). Azure deployment names are
        operator chosen but conventionally contain the model id (e.g. ``gpt-5-mini``,
        ``o3-prod``); when the guess is wrong, :meth:`_create_resilient` corrects it from the
        API's 400 response. The ``gpt-5-chat*`` variants are ordinary chat models (they accept
        a custom ``temperature`` and reject ``reasoning_effort``), so a deployment naming one is
        carved out before the gpt-5 reasoning match.
        """
        name = deployment.lower()
        if "gpt-5-chat" in name:
            return False
        return name.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in name

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
            StageError: If credentials/endpoint are missing, the deployment is unset, or the
                Azure OpenAI call fails (wrapped at the edge).
        """
        if not self._deployment:
            raise StageError(
                "enrich",
                "Azure OpenAI VLM requires a deployment name "
                "(set AZURE_OPENAI_VLM_DEPLOYMENT or pass deployment=).",
            )
        self._ensure_client()
        b64 = base64.b64encode(image).decode("ascii")
        # Azure OpenAI validates the declared data: URL content-type against the payload, so a
        # hard-coded image/png rejects valid JPEG/GIF/WebP figures (e.g. images extracted from
        # PDFs) with HTTP 400. Sniff the MIME type from the magic bytes (mirrors gpt4o.py).
        data_url = f"data:{_sniff_mime(image)};base64,{b64}"
        # gpt-5/o-series reasoning deployments (gpt-5 is multimodal) reject a custom
        # ``temperature`` on Chat Completions; sending one returns HTTP 400. Only forward
        # ``temperature`` for non-reasoning deployments and use ``reasoning_effort="minimal"``
        # for reasoning ones (mirrors the LLM adapters). Deployment names are opaque, so
        # :meth:`_create_resilient` corrects a wrong guess from the API's 400.
        kwargs: dict[str, Any] = {
            "model": self._deployment,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        if not self._is_reasoning_model(self._deployment):
            kwargs["temperature"] = 0.0  # determinism (coding-standards §1.4)
        else:
            kwargs["reasoning_effort"] = "minimal"
        resp = self._create_resilient(kwargs)
        # Convert vendor response -> core str AT THE EDGE. No openai type leaks upward. Guard
        # against an empty choices list (content-filtered responses) so we never raise IndexError.
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        return choices[0].message.content or ""

    def _create_resilient(self, kwargs: dict[str, Any]) -> Any:
        """Call ``chat.completions.create`` and recover from an unsupported-parameter 400.

        Azure addresses a model by an opaque *deployment* name, so the reasoning/non-reasoning
        guess in :meth:`_is_reasoning_model` can be wrong. When the deployed model rejects a
        parameter — gpt-5/o-series reject a custom ``temperature``; non-reasoning models reject
        ``reasoning_effort`` — we drop the offending parameter (swapping in its counterpart) and
        retry, rather than surfacing a confusing HTTP 400. At most two corrections are ever needed
        (temperature ⇄ reasoning_effort). Any other failure is converted to :class:`StageError`
        at the edge so the vendor exception never escapes this adapter (coding-standards §6.2).
        """
        client = self._ensure_client()
        for _ in range(3):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 — convert any vendor failure at the edge
                bad = _unsupported_param(exc)
                if bad == "temperature" and "temperature" in kwargs:
                    del kwargs["temperature"]
                    kwargs.setdefault("reasoning_effort", "minimal")
                elif bad == "reasoning_effort" and "reasoning_effort" in kwargs:
                    del kwargs["reasoning_effort"]
                    kwargs.setdefault("temperature", 0.0)
                else:
                    raise StageError("enrich", f"{self.name} (vlm) failed: {exc}") from exc
        # pragma: no cover - exhausted both corrections without success
        raise StageError("enrich", "Azure OpenAI VLM request did not complete.")
