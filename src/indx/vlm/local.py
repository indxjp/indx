"""LocalVLM — describe images via a locally-served, OpenAI-compatible VLM endpoint.

Requires the ``vlm-local`` extra (``pip install indx[vlm-local]``), which pulls ``httpx``.
The vendor SDK is imported lazily inside the method that needs it so importing this module
on a clean install is always safe (coding-standards §2/§6.3): the dependency gate runs in
``__init__``, raising :class:`~indx.errors.MissingExtraError` when the extra is absent.

The adapter POSTs a base64 data URI of the image bytes to an OpenAI-compatible
``/chat/completions`` endpoint and returns only the description ``str`` extracted at the
edge — no vendor object ever escapes into ``core/`` (coding-standards §6.2).
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra
from indx.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx  # type: ignore[import-not-found, unused-ignore]  # optional extra: vlm-local

_SLOT = "vlm"
_NAME = "local"
_EXTRA = "vlm-local"

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_PROMPT = "Describe this image in detail."

_logger = get_logger(__name__)


class LocalVLM:
    """Adapter for a local OpenAI-compatible VLM HTTP endpoint (satisfies ``VLM``).

    Installed via ``indx[vlm-local]``. Talks to a local server (e.g. vLLM, llama.cpp,
    Ollama's OpenAI shim) over ``httpx``. The base URL is taken from the ``base_url``
    kwarg, else the ``INDX_VLM_BASE_URL`` env var, else ``http://localhost:8000/v1``.
    No data leaves the host unless the configured endpoint is remote.

    Attributes:
        name: Registry name (``"local"``).
        model: The model identifier sent to the endpoint.
    """

    name = "local"

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialise the adapter and verify the optional extra is installed.

        Args:
            model: Model identifier for the endpoint. Defaults to ``"local-vlm"``.
            base_url: OpenAI-compatible base URL. Falls back to ``INDX_VLM_BASE_URL``
                then ``http://localhost:8000/v1``.
            timeout: Per-request timeout in seconds.

        Raises:
            MissingExtraError: If the ``vlm-local`` extra (``httpx``) is not installed.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "httpx")
        self.model = model or "local-vlm"
        resolved = base_url or os.environ.get("INDX_VLM_BASE_URL") or _DEFAULT_BASE_URL
        self._base_url = resolved.rstrip("/")
        self._timeout = timeout
        self._client: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_client(self) -> httpx.Client:
        """Return a lazily-constructed ``httpx.Client`` (vendor import inside)."""
        if self._client is None:
            import httpx  # optional extra: vlm-local (lazy, resolved via TYPE_CHECKING)

            self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout)
        return self._client

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image by POSTing it to the local VLM endpoint.

        The image bytes are base64-encoded into a ``data:`` URI and sent as an
        OpenAI vision content part. Only the extracted caption ``str`` is returned.

        Args:
            image: Raw image bytes (e.g. a figure captured from a parsed document).
            prompt: Optional instruction guiding the description. Defaults to a
                generic "describe this image" prompt.

        Returns:
            The model's description, or an empty string if the response carried none.

        Raises:
            StageError: If the endpoint returns an error or an unexpected payload.
        """
        client = self._ensure_client()
        b64 = base64.b64encode(image).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,  # determinism (coding-standards §1.4)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _DEFAULT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        }

        try:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # convert vendor/transport errors at the edge
            # Never echo the base URL host credentials or payload bytes into the message.
            _logger.warning("local VLM request failed: %s", type(exc).__name__)
            raise StageError("enrich", f"local VLM request failed: {exc}") from exc

        try:
            choices = data["choices"]
            message = choices[0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise StageError("enrich", "local VLM returned an unexpected response shape") from exc

        # Convert vendor JSON → core str AT THE EDGE.
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # some servers return content-part lists
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "".join(parts)
        return ""
