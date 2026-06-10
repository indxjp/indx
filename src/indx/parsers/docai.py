"""DocumentAIParser — cloud parser backend. Requires the ``gcp`` extra.

Google Document AI is a *hosted* service: it uploads the document to a processor that was
created out-of-band (console/API) and authenticates through Google ADC, so it is opt-in only
and never on the default path (reference/05-parsers.md). The vendor package is lazy-imported
inside ``__init__``/``parse`` so merely importing this module stays cheap and safe on a light
install; the ``require_extra`` gate turns an absent extra into a friendly
:class:`~indx.errors.MissingExtraError` at construction time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from google.cloud.documentai import (  # type: ignore[import-not-found]  # optional extra: gcp
        DocumentProcessorServiceClient,
    )

# Map a file suffix to the MIME type Document AI expects. Unknown suffixes fall back to PDF,
# the most common document input for the OCR processor.
_MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
_DEFAULT_MIME = "application/pdf"


def _split_paragraphs(text: str) -> list[str]:
    """Split extracted text into non-empty, stripped paragraphs on blank lines.

    Mirrors :class:`~indx.parsers.plaintext.PlainTextParser` so block segmentation stays
    consistent across parsers.

    Args:
        text: The full document text returned by Document AI.

    Returns:
        The ordered list of non-empty paragraph strings.
    """
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]


class DocumentAIParser:
    """Parser backend that uses Google Document AI (cloud) to extract document text.

    The adapter reads ``project``/``location`` from config (falling back to
    ``GOOGLE_CLOUD_PROJECT``/``GOOGLE_CLOUD_LOCATION``) and authenticates via Google ADC — no
    credentials are ever read or logged here. The processor is referenced by ``processor_id``
    and must be created out-of-band. No vendor type is allowed to escape this adapter
    (reference/05-parsers.md).
    """

    name = "docai"
    version = "1"

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        processor_id: str | None = None,
    ) -> None:
        """Construct the adapter and verify the optional extra is installed.

        Args:
            project: The Google Cloud project id. Falls back to ``GOOGLE_CLOUD_PROJECT``.
            location: The Document AI multi-region, ``"us"`` or ``"eu"`` (NOT a compute region
                like ``us-central1``). Falls back to ``GOOGLE_CLOUD_LOCATION``.
            processor_id: The id of a pre-created Document AI processor (e.g. an OCR processor).

        Raises:
            MissingExtraError: If the ``gcp`` extra is not installed.
        """
        require_extra("parser", "docai", "gcp", "google.cloud.documentai")
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us"
        self._processor_id = processor_id

    def _client(self) -> DocumentProcessorServiceClient:
        """Build the vendor Document AI client, lazily importing the SDK.

        Returns:
            A configured vendor ``DocumentProcessorServiceClient``. This object is a VENDOR
            type and must never escape this adapter.
        """
        # Lazy: imported only when we actually parse, never at module top level.
        # (Resolved for typing via the TYPE_CHECKING import above; optional extra: gcp.)
        from google.cloud import documentai  # type: ignore[import-not-found]

        return documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{self._location}-documentai.googleapis.com"}
        )

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one file into a :class:`~indx.core.parsed.ParsedDoc` via Document AI.

        Args:
            path: The local source file to upload and process.

        Returns:
            The normalized parsed document with extracted text flattened into blocks.

        Raises:
            StageError: If ``project``/``processor_id`` is missing or the call fails.
        """
        if not self._project:
            raise StageError(
                "parse",
                "docai needs a project: set GOOGLE_CLOUD_PROJECT or [parser.docai] project",
                path=str(path),
            )
        if not self._processor_id:
            raise StageError(
                "parse",
                "docai needs a processor_id: set [parser.docai] processor_id",
                path=str(path),
            )

        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), _DEFAULT_MIME)
        try:
            # Lazy import for the request types too — keeps module import SDK-free.
            from google.cloud import documentai

            client = self._client()
            name = client.processor_path(self._project, self._location, self._processor_id)
            result = client.process_document(
                request=documentai.ProcessRequest(
                    name=name,
                    raw_document=documentai.RawDocument(content=path.read_bytes(), mime_type=mime),
                )
            )
            text = result.document.text or ""
        except Exception as exc:  # vendor exception — translate at the edge
            raise StageError("parse", f"docai failed: {exc}", path=str(path)) from exc

        # Convert at the edge: the vendor Document dies here, only ParsedDoc escapes.
        blocks = [
            Block(kind="text", text=raw, order=order)
            for order, raw in enumerate(_split_paragraphs(text))
        ]
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )
