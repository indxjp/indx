"""VertexLLM — Vertex AI Gemini LLM. Requires the `gcp` extra (uses the google-genai SDK).

The Enrich stage talks to text models through the :class:`~indx.llm.base.LLM` protocol. This
adapter wraps the unified ``google-genai`` SDK's ``models.generate_content`` call in Vertex AI
mode. Per ``docs/coding-standards`` §2/§6, the heavy ``google-genai`` SDK is imported lazily
inside the client factory so a clean ``pip install indx`` never pays for it, and the
optional-extra gate (``require_extra``) runs at construction so selecting this backend without
``indx[gcp]`` fails with an actionable :class:`~indx.errors.MissingExtraError`.

Auth is Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``, an attached
service account, or ``gcloud auth application-default login``); this adapter never reads or
logs API keys. Project/location come from kwargs or ``GOOGLE_CLOUD_PROJECT`` /
``GOOGLE_CLOUD_LOCATION`` (coding-standards §2/§5).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from google import genai  # type: ignore[import-not-found]  # optional extra: gcp

logger = logging.getLogger(__name__)

_SLOT = "llm"
_NAME = "vertex"
_EXTRA = "gcp"

# Never hard-pin a "latest": model ids carry retirement dates (gemini-2.5-* retire 2026-10-16).
_DEFAULT_MODEL = "gemini-2.5-flash"


class VertexLLM:
    """Vertex AI Gemini adapter (satisfies the ``LLM`` protocol structurally).

    Installed via ``indx[gcp]``. The ``google-genai`` package is imported lazily inside the
    client factory so a clean ``pip install indx`` never pays for it (coding-standards §2).
    Credentials come from Application Default Credentials and are never logged or serialized.

    Attributes:
        name: Registry name for this backend (``"vertex"``).
        model: The Gemini model identifier in use.
    """

    name = "vertex"

    def __init__(
        self,
        model: str | None = None,
        *,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        """Configure the Vertex AI Gemini adapter.

        Args:
            model: Gemini model identifier. Defaults to ``gemini-2.5-flash`` when ``None``.
            project: GCP project id. Defaults to the ``GOOGLE_CLOUD_PROJECT`` environment
                variable.
            location: GCP location/region. Defaults to the ``GOOGLE_CLOUD_LOCATION``
                environment variable.
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "google.genai")
        self.model = model or _DEFAULT_MODEL
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        self._client: genai.Client | None = None

    def _ensure_client(self) -> genai.Client:
        """Lazily build and cache the Vertex AI genai client.

        The ``google-genai`` import is performed here (not at module top level) so importing
        this module is always safe on a light-core install.

        Returns:
            The configured ``google.genai.Client`` in Vertex AI mode.
        """
        if self._client is None:
            from google import genai  # optional extra: gcp (lazy; see TYPE_CHECKING import)

            logger.debug(
                "creating Vertex AI genai client project=%s location=%s model=%s",
                self._project,
                self._location,
                self.model,
            )
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
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
        """Generate a completion for ``prompt`` against the configured Gemini model.

        Args:
            prompt: The user prompt.
            system: Optional system instruction.
            max_tokens: Maximum number of output tokens to generate.
            temperature: Sampling temperature; defaults to ``0.0`` for determinism.

        Returns:
            The model's text content, or an empty string if none was returned.

        Raises:
            StageError: If the underlying Vertex AI call fails.
        """
        from google.genai import types  # type: ignore[import-not-found]  # optional extra: gcp

        client = self._ensure_client()
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — wrap any vendor failure at the edge
            raise StageError(
                "enrich",
                f"Vertex AI generate_content failed for model {self.model!r}: {exc}",
            ) from exc
        # Convert the vendor response → core str AT THE EDGE. No genai type leaks upward.
        return resp.text or ""
