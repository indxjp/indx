"""AnthropicLLM — Claude enrichment LLM (Messages API). Requires the ``anthropic`` extra.

Thin adapter over the Anthropic Python SDK's ``client.messages.create`` that satisfies the
:class:`~indx.llm.base.LLM` protocol structurally. The vendor SDK is imported lazily inside the
adapter so importing this module on a clean ``pip install indx`` is always safe; selecting the
slot without the extra raises :class:`~indx.errors.MissingExtraError` at construction.

The API key is read from the ``ANTHROPIC_API_KEY`` environment variable by the SDK — it is never
hard-coded, logged, or serialized (coding-standards §8/§13).
"""

from __future__ import annotations

import logging
from typing import Any

from indx.utils.lazy import require_extra

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "anthropic"
_EXTRA = "anthropic"
_DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicLLM:
    """Anthropic Messages API adapter for the Enrich stage.

    Installed via ``indx[anthropic]``. Talks to Claude through
    ``client.messages.create(...)`` and returns the concatenated text blocks of the response.
    A provider ``anthropic.Message`` is converted to a core ``str`` at the adapter edge and never
    leaks into ``core/`` (coding-standards §6.2).

    Attributes:
        name: Registry key for this implementation (``"anthropic"``).
        model: The Claude model ID used for completions (e.g. ``"claude-haiku-4-5"``).
    """

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        """Construct the adapter and fail fast if the extra is missing.

        Args:
            model: Claude model ID (registry-supplied ``:model`` suffix). Defaults to
                ``"claude-haiku-4-5"`` — a small, cheap model suited to short Enrich outputs;
                the user overrides it via config, so no choice is hard-coded that can't be changed.

        Raises:
            MissingExtraError: If the ``anthropic`` extra is not installed.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "anthropic")
        self.model = model or _DEFAULT_MODEL
        # Any: the vendor anthropic.Anthropic client type is an optional extra absent at
        # type-check time; it is confined to this adapter, never returned into core/ (§6.2).
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Lazily construct the Anthropic client (reads ``ANTHROPIC_API_KEY`` from the env).

        Returns:
            The vendor ``anthropic.Anthropic`` client (typed ``Any`` — the SDK is an optional
            extra absent at type-check time and never escapes this adapter).
        """
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]  # optional extra: anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` and return the joined text blocks.

        Args:
            prompt: The user prompt to complete.
            system: Optional system instruction (omitted from the request when ``None``).
            max_tokens: Maximum number of tokens to generate (required by the Messages API).
            temperature: Sampling temperature; defaults to ``0.0`` for deterministic-leaning
                output (coding-standards §1.4).

        Returns:
            The concatenation of every ``text`` block in the response. Non-text blocks (e.g.
            ``thinking``) are skipped.
        """
        client = self._ensure_client()
        # Build kwargs so `system` is only sent when provided — the Messages API rejects a
        # `system=None` and treats an absent key as "no system prompt".
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        logger.debug("anthropic complete: model=%s max_tokens=%d", self.model, max_tokens)

        response = client.messages.create(**kwargs)
        # Convert vendor response -> core str AT THE EDGE: walk content blocks and join only the
        # text blocks. The anthropic.Message never escapes this method.
        return "".join(block.text for block in response.content if block.type == "text")
