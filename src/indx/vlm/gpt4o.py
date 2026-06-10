"""GPT4oVLM — OpenAI vision adapter (default model ``gpt-4o``). Requires the `openai` extra.

The ``openai`` SDK is imported lazily inside :meth:`GPT4oVLM._ensure_client` so a clean
``pip install indx`` never pays for it (coding-standards §2). Construction calls
``require_extra`` first, so selecting this backend without the extra fails fast with the
friendly :class:`~indx.errors.MissingExtraError` (file-architecture §5).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from openai import (  # type: ignore[import-not-found]  # optional extra: openai
        OpenAI,
    )

_SLOT = "vlm"
_NAME = "gpt4o"
_EXTRA = "openai"

_DEFAULT_PROMPT = "Describe this image in detail."

logger = logging.getLogger(__name__)


def _sniff_mime(image: bytes) -> str:
    """Sniff the image MIME type from magic bytes, defaulting to ``image/png``.

    OpenAI validates the declared ``data:`` URL content-type against the payload, so a
    hard-coded ``image/png`` rejects valid JPEG/GIF/WebP figures (e.g. images extracted
    from PDFs). Recognises PNG, JPEG, GIF and WebP; anything else falls back to PNG.
    """
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class GPT4oVLM:
    """OpenAI multimodal chat adapter that describes images (satisfies the VLM protocol).

    Installed via ``indx[openai]``. Image bytes are encoded as a base64 ``data:`` URL and
    sent through the Chat Completions vision content shape; only the extracted caption
    string is returned. The ``openai.ChatCompletion`` response never escapes this adapter
    (coding-standards §6.2).
    """

    name = "gpt4o"

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return ``True`` for models that reject custom ``temperature`` on Chat Completions.

        The gpt-5 *reasoning* family and o-series models only accept their default temperature;
        sending another value returns HTTP 400. The ``gpt-5-chat*`` variants accept a custom
        ``temperature`` and reject ``reasoning_effort``, so they are carved out before the
        ``gpt-5`` prefix test (mirrors :meth:`OpenAILLM._is_reasoning_model`).
        """
        name = model.lower()
        if name.startswith("gpt-5-chat"):
            return False
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def __init__(self, model: str | None = None) -> None:
        """Initialize the adapter.

        Args:
            model: The OpenAI vision-capable model id. Defaults to ``gpt-4o``.

        Raises:
            MissingExtraError: If the ``openai`` extra is not installed.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "openai")
        self.model = model or "gpt-4o"
        self._client: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_client(self) -> OpenAI:
        """Lazily build and cache the OpenAI client.

        The API key is read from the ``OPENAI_API_KEY`` environment variable by the SDK;
        it is never hard-coded or logged (coding-standards §8).

        Returns:
            The cached :class:`openai.OpenAI` client.
        """
        if self._client is None:
            from openai import OpenAI  # optional extra: openai

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.debug("vlm backend=%s model=%s key=***redacted***", _NAME, self.model)
        return self._client

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image using the OpenAI vision chat model.

        Args:
            image: Raw image bytes (e.g. PNG/JPEG) to describe.
            prompt: Optional instruction guiding the description. Defaults to a generic
                "describe this image" prompt.

        Returns:
            The model's description text, or an empty string if the model returned none.

        Raises:
            StageError: If the OpenAI call fails (wrapped at the edge).
        """
        client = self._ensure_client()
        b64 = base64.b64encode(image).decode("ascii")
        data_url = f"data:{_sniff_mime(image)};base64,{b64}"
        # gpt-5/o-series reasoning models (gpt-5 is multimodal) reject a custom ``temperature``
        # on Chat Completions; sending one returns HTTP 400. Only forward ``temperature`` for
        # non-reasoning models (e.g. the default gpt-4o) and use ``reasoning_effort="minimal"``
        # for reasoning ones (mirrors the LLM adapters).
        kwargs: dict[str, Any] = {
            "model": self.model,
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
        if not self._is_reasoning_model(self.model):
            kwargs["temperature"] = 0.0  # determinism (coding-standards §1.4)
        else:
            kwargs["reasoning_effort"] = "minimal"
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — convert any vendor failure at the edge
            raise StageError("enrich", f"{self.name} (vlm) failed: {exc}") from exc
        # Convert vendor response -> core str AT THE EDGE. Guard against an empty
        # choices list (content-filtered responses) so we never raise IndexError.
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        return choices[0].message.content or ""
