"""VertexVLM — Vertex AI Gemini multimodal adapter (default ``gemini-2.5-flash``).

Requires the `gcp` extra. The ``google-genai`` SDK is imported lazily inside
:meth:`VertexVLM._ensure_client` so a clean ``pip install indx`` never pays for it
(coding-standards §2). Construction calls ``require_extra`` first, so selecting this
backend without the extra fails fast with the friendly
:class:`~indx.errors.MissingExtraError` (file-architecture §5).

This adapter reuses the same ``genai.Client`` / model wiring as :class:`VertexLLM`,
sending the image as an inline part alongside the prompt and returning only the caption
string; the vendor response never escapes the adapter (coding-standards §6.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra
from indx.vlm.gpt4o import _sniff_mime

if TYPE_CHECKING:
    from google.genai import (  # type: ignore[import-not-found]  # optional extra: gcp
        Client,
    )

_SLOT = "vlm"
_NAME = "vertex"
_EXTRA = "gcp"

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_PROMPT = "Describe this image in detail."

logger = logging.getLogger(__name__)


class VertexVLM:
    """Vertex AI Gemini multimodal adapter that describes images (satisfies VLM).

    Installed via ``indx[gcp]``. Auth uses Application Default Credentials in Vertex mode
    (``GOOGLE_APPLICATION_CREDENTIALS`` / attached SA / ``gcloud auth application-default
    login``); no API key is ever read or logged (coding-standards §8). Project and location
    come from config kwargs or the ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``
    environment variables.

    Attributes:
        name: The registry key used in config and on the CLI.
        model: The Gemini model id (configurable; never hard-pinned — model IDs carry hard
            retirement dates).
    """

    name = _NAME

    def __init__(
        self,
        model: str | None = None,
        *,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The Vertex Gemini model id. Defaults to ``gemini-2.5-flash``.
            project: GCP project id. Falls back to the SDK's environment/ADC resolution
                (``GOOGLE_CLOUD_PROJECT``) when ``None``.
            location: GCP location. Falls back to the SDK's environment/ADC resolution
                (``GOOGLE_CLOUD_LOCATION``) when ``None``.

        Raises:
            MissingExtraError: If the ``gcp`` extra is not installed.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "google.genai")
        self.model = model or _DEFAULT_MODEL
        self._project = project
        self._location = location
        self._client: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_client(self) -> Client:
        """Lazily build and cache the Vertex-mode ``genai`` client.

        Credentials are resolved by the SDK's own Application Default Credentials chain;
        they are never hard-coded or logged (coding-standards §8).

        Returns:
            The cached :class:`google.genai.Client`.
        """
        if self._client is None:
            from google import (  # type: ignore[import-not-found]  # optional extra: gcp
                genai,
            )

            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
            logger.debug("vlm backend=%s model=%s", _NAME, self.model)
        return self._client

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image using the Vertex Gemini multimodal model.

        Args:
            image: Raw image bytes (e.g. PNG/JPEG) to describe.
            prompt: Optional instruction guiding the description. Defaults to a generic
                "describe this image" prompt.

        Returns:
            The model's description text, or an empty string if the model returned none.

        Raises:
            StageError: If the underlying Vertex AI call fails (wrapped at the edge).
        """
        from google.genai import types  # optional extra: gcp

        client = self._ensure_client()
        # Gemini decodes the inline part using the declared mime_type; sniff it from the
        # magic bytes so JPEG/GIF/WebP figures (e.g. extracted from PDFs) are not mislabeled.
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=[
                    prompt or _DEFAULT_PROMPT,
                    types.Part.from_bytes(data=image, mime_type=_sniff_mime(image)),
                ],
            )
        except Exception as exc:  # noqa: BLE001 — convert any vendor failure at the edge
            raise StageError("enrich", f"{self.name} (vlm) failed: {exc}") from exc
        # Convert vendor response -> core str AT THE EDGE.
        return resp.text or ""
