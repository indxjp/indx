"""VLLMClient — talk to a vLLM OpenAI-compatible endpoint via the ``openai`` SDK.

vLLM serves an OpenAI-compatible HTTP API, so there is no dedicated vLLM SDK: indx reuses
the ``openai`` client pointed at the vLLM ``base_url`` (default ``http://localhost:8000/v1``)
with a throwaway key (reference/06-llm.md "vLLM — reuse the OpenAI adapter"). Requires the
`vllm` extra; importing this module is always safe and the gate fires at construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import OpenAI  # type: ignore[import-not-found]  # optional extra: vllm

_SLOT = "llm"
_NAME = "vllm"
_EXTRA = "vllm"
_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "qwen2.5"

logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM adapter satisfying the :class:`~indx.llm.base.LLM` protocol structurally.

    Installed via ``indx[vllm]``. The ``openai`` package is imported lazily inside the
    client factory so a clean ``pip install indx`` never pays for it (coding-standards §2).
    Talks to a vLLM server's OpenAI-compatible Chat Completions endpoint; the API key is a
    throwaway (vLLM does not authenticate by default) read from ``OPENAI_API_KEY`` and never
    logged (coding-standards §8/§13).
    """

    name = "vllm"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Construct the adapter and fail fast if the ``vllm`` extra is absent.

        Args:
            model: The served model name (e.g. ``"qwen2.5"``). The registry splits a
                ``vllm:<model>`` spec and passes the suffix here; defaults to ``"qwen2.5"``.
            base_url: The vLLM OpenAI-compatible endpoint. Defaults to
                ``http://localhost:8000/v1``.
            api_key: Throwaway key for the vLLM server. Defaults to the ``OPENAI_API_KEY``
                env var, falling back to ``"EMPTY"`` (vLLM does not authenticate by default).

        Raises:
            MissingExtraError: If the ``vllm`` extra (the ``openai`` SDK) is not installed.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        import os

        self.model = model or _DEFAULT_MODEL
        self._base_url = base_url or _DEFAULT_BASE_URL
        # vLLM ignores the key by default; read from env, never hard-code a secret.
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        """Lazily build and cache the OpenAI client pointed at the vLLM endpoint."""
        if self._client is None:
            from openai import OpenAI  # optional extra: vllm

            logger.debug("vllm backend base_url=%s model=%s", self._base_url, self.model)
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` via the vLLM Chat Completions endpoint.

        Args:
            prompt: The user prompt to complete.
            system: Optional system instruction prepended as a ``system`` message.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature; defaults to ``0.0`` for determinism (§1.4).

        Returns:
            The generated text, or an empty string if the model returned no content.
        """
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._ensure_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Convert vendor response -> core str AT THE EDGE; no openai type escapes the adapter.
        return resp.choices[0].message.content or ""
