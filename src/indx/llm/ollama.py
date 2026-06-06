"""OllamaLLM — local enrichment LLM (``ollama:qwen2.5``). Requires the `ollama` extra.

This is indx's local-profile LLM: it talks to a local Ollama daemon (default
``http://localhost:11434``), needs no API key, and sends nothing to the cloud
(technology-selections §7, coding-standards §13). The ``ollama`` package is imported
lazily inside the method that needs it so a clean ``pip install indx`` never pays for it
(coding-standards §2); selecting this slot without the extra raises the friendly
:class:`~indx.errors.MissingExtraError` at construction (file-architecture §5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_SLOT, _NAME, _EXTRA = "llm", "ollama", "ollama"
_DEFAULT_MODEL = "qwen2.5"


class OllamaLLM:
    """Local Ollama adapter satisfying the :class:`~indx.llm.base.LLM` protocol.

    Default model is ``qwen2.5``. No API key is used; the adapter talks to a local
    Ollama daemon at ``http://localhost:11434`` unless ``host`` overrides it. The vendor
    ``ollama`` response object is converted to a plain ``str`` at the adapter edge and
    never leaks into ``core/`` (coding-standards §6.2).
    """

    name = _NAME

    def __init__(self, model: str | None = None, *, host: str | None = None) -> None:
        """Initialise the adapter.

        Args:
            model: Ollama model tag to chat against. Defaults to ``qwen2.5``.
            host: Optional Ollama daemon URL (e.g. ``http://localhost:11434``). When
                ``None`` the ``ollama`` client uses its own default.

        Raises:
            MissingExtraError: If the ``ollama`` extra is not installed.
        """
        # Cheap find_spec gate: selecting an uninstalled backend fails fast with the
        # pip hint instead of an ImportError deep inside the vendor package.
        require_extra(_SLOT, _NAME, _EXTRA, "ollama")
        self.model = model or _DEFAULT_MODEL
        self._host = host
        self._client: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_client(self) -> Any:
        """Lazily build and cache the Ollama client.

        Returns:
            The vendor ``ollama.Client`` instance (kept as ``Any`` so the vendor type
            never escapes this module).
        """
        if self._client is None:
            from ollama import Client  # type: ignore[import-not-found]  # optional extra: ollama

            self._client = Client(host=self._host) if self._host else Client()
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt`` via the local Ollama daemon.

        Args:
            prompt: The user prompt to complete.
            system: Optional system instruction prepended as a ``system`` message.
            max_tokens: Upper bound on generated tokens (mapped to Ollama's
                ``num_predict`` option).
            temperature: Sampling temperature; defaults to ``0.0`` for determinism
                (coding-standards §1.4).

        Returns:
            The assistant's reply text, or ``""`` if the model returned no content.

        Raises:
            StageError: If the Ollama daemon call fails (e.g. the daemon is not running
                or the model is not pulled).
        """
        messages: list[Mapping[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._ensure_client().chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
        except Exception as exc:  # vendor/httpx error → typed, actionable stage failure
            raise StageError(
                "enrich",
                f"Ollama chat failed for model '{self.model}'. "
                "Is the daemon running (ollama serve) and the model pulled "
                f"(ollama pull {self.model})? {exc}",
            ) from exc

        # Convert vendor response → core str AT THE EDGE. Nothing above this line ever
        # sees an ollama.ChatResponse.
        content = response.message.content
        return content if content else ""
