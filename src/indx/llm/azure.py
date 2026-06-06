"""AzureOpenAILLM — Azure-hosted OpenAI LLM. Requires the `azure` extra (uses the openai SDK).

Mirrors :class:`indx.llm.openai.OpenAILLM` but targets an Azure OpenAI deployment via
``openai.AzureOpenAI``. Credentials and endpoint are read from kwargs or, when omitted, from
the standard Azure environment variables. The ``openai`` SDK is imported lazily so importing
this module on a clean install is always safe (coding-standards §2/§6.3).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import AzureOpenAI  # type: ignore[import-not-found]  # optional extra: azure

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "azure"
_EXTRA = "azure"

_DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAILLM:
    """Azure OpenAI Chat Completions adapter (satisfies the ``LLM`` protocol structurally).

    Installed via ``indx[azure]``. The ``openai`` package is imported lazily inside the
    client factory so a clean ``pip install indx`` never pays for it (coding-standards §2).
    Secrets are read from the environment and are never logged or serialized (§8).
    """

    name = "azure"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment: str | None = None,
    ) -> None:
        """Configure the Azure OpenAI adapter.

        Args:
            model: Deployment name to invoke. Falls back to ``deployment`` and then to the
                ``AZURE_OPENAI_DEPLOYMENT`` environment variable.
            api_key: Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY``.
            azure_endpoint: Resource endpoint URL. Defaults to ``AZURE_OPENAI_ENDPOINT``.
            api_version: Azure REST API version. Defaults to ``AZURE_OPENAI_API_VERSION``
                or a pinned recent version.
            deployment: Alias for ``model`` (the Azure deployment name).
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        self.model = model or deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or _DEFAULT_API_VERSION
        )
        self._client: AzureOpenAI | None = None

    def _ensure_client(self) -> AzureOpenAI:
        """Lazily build and cache the Azure OpenAI client.

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
                "creating Azure OpenAI client endpoint=%s api_version=%s deployment=%s",
                self._azure_endpoint,
                self._api_version,
                self.model,
            )
            self._client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` against the configured deployment.

        Args:
            prompt: The user prompt.
            system: Optional system instruction prepended as a system message.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature; defaults to ``0.0`` for determinism.

        Returns:
            The assistant's text content, or an empty string if none was returned.

        Raises:
            StageError: If credentials/endpoint are missing or the deployment is unset.
        """
        if not self.model:
            raise StageError(
                "enrich",
                "Azure OpenAI requires a deployment name "
                "(set AZURE_OPENAI_DEPLOYMENT or pass model=/deployment=).",
            )
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # The gpt-5 family and o-series reasoning models reject the legacy ``max_tokens``
        # parameter (they require ``max_completion_tokens``) and only accept their default
        # temperature on Chat Completions; sending either returns HTTP 400. Azure addresses a
        # model by an opaque *deployment* name, so we detect the family from that name (which
        # conventionally embeds the model id). This mirrors the OpenAI adapter's handling.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if not self._is_reasoning_model(self.model):
            kwargs["temperature"] = temperature
        else:
            # Reasoning deployments spend the token budget on hidden reasoning first; cap it with
            # ``reasoning_effort="minimal"`` so short enrichment prompts return visible content
            # instead of an empty string (mirrors the OpenAI adapter).
            kwargs["reasoning_effort"] = "minimal"
        resp = self._create_resilient(kwargs, temperature)
        # Convert the vendor response → core str AT THE EDGE. No openai type leaks upward.
        return resp.choices[0].message.content or ""

    def _create_resilient(self, kwargs: dict[str, Any], temperature: float) -> Any:
        """Call ``chat.completions.create`` and recover from an unsupported-parameter 400.

        Azure addresses a model by an opaque *deployment* name, so the reasoning/non-reasoning
        guess in :meth:`_is_reasoning_model` can be wrong. When the deployed model rejects a
        parameter — gpt-5/o-series reject a custom ``temperature``; non-reasoning models reject
        ``reasoning_effort`` — we drop the offending parameter (swapping in its counterpart) and
        retry, rather than surfacing a confusing HTTP 400 to the user. At most two corrections
        are ever needed (temperature ⇄ reasoning_effort).
        """
        client = self._ensure_client()
        last: Exception | None = None
        for _ in range(3):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised unless it is a known param 400
                bad = _unsupported_param(exc)
                if bad == "temperature" and "temperature" in kwargs:
                    del kwargs["temperature"]
                    kwargs.setdefault("reasoning_effort", "minimal")
                elif bad == "reasoning_effort" and "reasoning_effort" in kwargs:
                    del kwargs["reasoning_effort"]
                    kwargs.setdefault("temperature", temperature)
                else:
                    raise
                last = exc
        if last is not None:  # pragma: no cover - exhausted both corrections
            raise last
        raise StageError("enrich", "Azure OpenAI request did not complete.")

    @staticmethod
    def _is_reasoning_model(deployment: str) -> bool:
        """Return ``True`` if the deployment name denotes a gpt-5/o-series reasoning model.

        These reject custom ``temperature`` and the legacy ``max_tokens`` on Chat Completions
        (mirrors :meth:`OpenAILLM._is_reasoning_model`). Azure deployment names are operator
        chosen but conventionally contain the model id (e.g. ``gpt-5-mini``, ``o3-prod``); when
        the guess is wrong, :meth:`_create_resilient` corrects it from the API's 400 response.
        """
        name = deployment.lower()
        return name.startswith(("gpt-5", "o1", "o3", "o4")) or "gpt-5" in name


def _unsupported_param(exc: Exception) -> str | None:
    """Name the request parameter an OpenAI/Azure 400 says is unsupported, or ``None``.

    Matches the SDK's ``BadRequestError`` (status 400) whose message flags ``temperature`` or
    ``reasoning_effort`` as unsupported, without importing the optional ``openai`` types at
    module top (the SDK is an extra). Used to drive the retry in :meth:`_create_resilient`.
    """
    if type(exc).__name__ != "BadRequestError" and getattr(exc, "status_code", None) != 400:
        return None
    msg = str(getattr(exc, "message", None) or exc).lower()
    if "support" not in msg:
        return None
    if "temperature" in msg:
        return "temperature"
    if "reasoning_effort" in msg:
        return "reasoning_effort"
    return None
