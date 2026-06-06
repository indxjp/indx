"""LiteLLMClient — reach *any* LLM provider through one adapter (``litellm:<provider/model>``).

LiteLLM normalizes 100+ providers behind a single ``completion`` call, so one indx backend
covers on-prem runtimes (Ollama, vLLM) **and** every managed cloud (AWS Bedrock, Azure OpenAI,
GCP Vertex, Anthropic, OpenAI, …). Select the provider with LiteLLM's ``provider/model`` string
via the registry's ``:model`` suffix::

    indx ./docs --out ./out --llm litellm:bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    indx ./docs --out ./out --llm litellm:azure/my-gpt-4o-deployment
    indx ./docs --out ./out --llm litellm:vertex_ai/gemini-1.5-pro
    indx ./docs --out ./out --llm litellm:ollama/llama3            # on-prem, no API key

Credentials come from each provider's standard environment variables (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, the AWS chain, ``AZURE_*``, Google ADC, …) — read by LiteLLM, never
stored on this object. An on-prem base URL can be set with ``LITELLM_API_BASE``.

The heavy ``litellm`` SDK is imported lazily inside the call path and gated by
:func:`~indx.utils.lazy.require_extra`, so importing this module is always safe on a light-core
install (file-architecture §5, coding-standards §6.3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from indx.utils.lazy import require_extra

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "litellm"
_EXTRA = "litellm"
_DEFAULT_MODEL = "gpt-4o-mini"


class LiteLLMClient:
    """LiteLLM-backed enrichment LLM (satisfies the ``LLM`` protocol structurally).

    Installed via ``indx[litellm]``. The model is a LiteLLM ``provider/model`` string; the
    default is ``gpt-4o-mini``. The vendor response is converted to ``str`` at the edge — no
    LiteLLM object ever escapes this adapter into core.

    Attributes:
        name: Registry name for this backend (``"litellm"``).
        model: The LiteLLM ``provider/model`` identifier in use.
    """

    name = "litellm"

    def __init__(self, model: str | None = None, *, api_base: str | None = None) -> None:
        """Construct the adapter, failing fast if the ``litellm`` extra is absent.

        Args:
            model: LiteLLM ``provider/model`` string. Defaults to ``gpt-4o-mini``.
            api_base: Optional base URL for self-hosted/on-prem endpoints. Falls back to the
                ``LITELLM_API_BASE`` environment variable.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "litellm")
        self.model = model or _DEFAULT_MODEL
        self.api_base = api_base or os.environ.get("LITELLM_API_BASE")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` and return the message text.

        ``drop_params=True`` lets LiteLLM silently drop any parameter the selected provider
        rejects (e.g. ``temperature`` on reasoning models), so one call works across every
        backend. Returns an empty string if the model produced no content.
        """
        import litellm  # type: ignore[import-not-found]  # optional extra: litellm

        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "drop_params": True,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        logger.debug("litellm.complete model=%s max_tokens=%d", self.model, max_tokens)

        response = litellm.completion(**kwargs)
        # Coerce at the adapter edge: no LiteLLM response object leaks into core.
        content = response.choices[0].message.content
        return content or ""
