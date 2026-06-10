"""OpenAILLM — default enrichment LLM (``openai:gpt-5-mini``). Requires the `openai` extra.

The Enrich stage talks to text models through the :class:`~indx.llm.base.LLM` protocol. This
adapter wraps the OpenAI Python SDK's ``chat.completions.create`` call. Per
``docs/coding-standards`` §2/§6, the heavy ``openai`` SDK is imported lazily inside the
client factory so a clean
``pip install indx`` never pays for it, and the optional-extra gate (``require_extra``) runs at
construction so selecting this backend without ``indx[openai]`` fails with an actionable
:class:`~indx.errors.MissingExtraError` instead of a deep ``ImportError``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import OpenAI  # type: ignore[import-not-found]  # optional extra: openai

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "openai"
_EXTRA = "openai"
_DEFAULT_MODEL = "gpt-5-mini"


class OpenAILLM:
    """OpenAI Chat Completions adapter (satisfies the ``LLM`` protocol structurally).

    Installed via ``indx[openai]``. The default model is ``gpt-5-mini`` and this is indx's
    default LLM backend. The API key is read from the ``OPENAI_API_KEY`` environment variable
    by the SDK; it is never hard-coded, logged, or stored on the instance.

    Attributes:
        name: Registry name for this backend (``"openai"``).
        model: The chat-completions model identifier in use.
    """

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        """Construct the adapter, failing fast if the ``openai`` extra is absent.

        Args:
            model: Chat model identifier. Defaults to ``gpt-5-mini`` when ``None``.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        self.model = model or _DEFAULT_MODEL
        # Any: holds the vendor ``openai.OpenAI`` client, which must never leak into core/.
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        """Lazily build and cache the OpenAI client.

        The ``openai`` import is performed here (not at module top level) so importing this
        module is always safe on a light-core install.

        Returns:
            The cached :class:`openai.OpenAI` client.
        """
        if self._client is None:
            from openai import OpenAI  # optional extra: openai

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return ``True`` for models that reject custom ``temperature`` on Chat Completions.

        The gpt-5 *reasoning* family and o-series models (e.g. ``o1``, ``o3``, ``o4-mini``) only
        accept their default temperature; sending another value returns HTTP 400. The
        ``gpt-5-chat*`` variants (``gpt-5-chat``, ``gpt-5-chat-latest``) are *not* reasoning
        models: they accept a custom ``temperature`` and reject ``reasoning_effort``, so they
        must be carved out before the ``gpt-5`` prefix test.
        """
        name = model.lower()
        if name.startswith("gpt-5-chat"):
            return False
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` and return the message text.

        Args:
            prompt: The user prompt to complete.
            system: Optional system instruction prepended as a ``system`` message.
            max_tokens: Maximum tokens to generate (sent as ``max_completion_tokens``).
            temperature: Sampling temperature. Only forwarded for models that accept a custom
                value; the gpt-5 family and o-series reasoning models reject anything other than
                their default and so ``temperature`` is omitted for them.

        Returns:
            The assistant message text, or an empty string if the model returned no content
            (including the content-filter edge case where the provider returns no choices).

        Raises:
            StageError: If the underlying OpenAI Chat Completions call fails.
        """
        client = self._ensure_client()
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # The gpt-5 family and o-series reasoning models reject the legacy ``max_tokens``
        # parameter (require ``max_completion_tokens``) and only accept their default
        # temperature on Chat Completions; sending either returns HTTP 400.
        forward_temperature = not self._is_reasoning_model(self.model)
        logger.debug(
            "openai.complete backend=%s model=%s max_completion_tokens=%d temperature=%s",
            _NAME,
            self.model,
            max_tokens,
            temperature if forward_temperature else "<model-default>",
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if forward_temperature:
            kwargs["temperature"] = temperature
        else:
            # Reasoning models spend ``max_completion_tokens`` on hidden reasoning *first*; with a
            # small cap (and short enrichment prompts) the budget can be exhausted before any
            # visible content, returning an empty string. ``reasoning_effort="minimal"`` keeps the
            # answer within a tight budget for the short, factual prompts the Enrich stage sends.
            kwargs["reasoning_effort"] = "minimal"
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — wrap any vendor failure at the edge
            raise StageError(
                "enrich",
                f"OpenAI chat.completions.create failed for model {self.model!r}: {exc}",
            ) from exc
        # Convert the vendor response -> core ``str`` AT THE EDGE: no openai.ChatCompletion
        # object ever escapes this adapter into core/. A provider returning zero choices
        # (e.g. an Azure/OpenAI-compatible content-filter block) must yield "" not IndexError.
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        content = choices[0].message.content
        return content or ""
