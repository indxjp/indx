"""DocumentIntelligenceParser — cloud parser backend. Requires the ``azure`` extra.

Azure AI Document Intelligence is a *hosted* OCR/layout service: it uploads the document and
needs an endpoint + API key, so it is opt-in only and never on the default path
(reference/05-parsers.md). The vendor SDK is lazy-imported inside ``parse`` so merely importing
this module stays cheap and safe on a light install; the ``require_extra`` gate turns an absent
extra into a friendly :class:`~indx.errors.MissingExtraError` at construction time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from azure.ai.documentintelligence import (  # type: ignore[import-not-found]  # optional extra: azure
        DocumentIntelligenceClient,
    )

_SLOT = "parser"
_NAME = "docintel"
_EXTRA = "azure"

_DEFAULT_MODEL_ID = "prebuilt-read"


def _split_paragraphs(text: str) -> list[str]:
    """Split rendered text into non-empty, stripped paragraphs on blank lines.

    Mirrors :class:`~indx.parsers.plaintext.PlainTextParser` so block segmentation stays
    consistent across text-producing parsers.

    Args:
        text: The full text returned by Document Intelligence (``result.content``).

    Returns:
        The ordered list of non-empty paragraph strings.
    """
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]


class DocumentIntelligenceParser:
    """Parser backend that uses Azure AI Document Intelligence to extract document text.

    Installed via ``indx[azure]``. The adapter reads ``AZURE_DOCUMENTINTELLIGENCE_ENDPOINT``
    and ``AZURE_DOCUMENTINTELLIGENCE_KEY`` from the environment (never hard-coded) and flattens
    the service's extracted text into :class:`~indx.core.parsed.Block` objects. No vendor type
    is allowed to escape this adapter (reference/05-parsers.md).
    """

    name = "docintel"
    version = "1"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
    ) -> None:
        """Construct the adapter and verify the optional extra is installed.

        Args:
            endpoint: Document Intelligence resource endpoint. Defaults to
                ``AZURE_DOCUMENTINTELLIGENCE_ENDPOINT``.
            key: Document Intelligence API key. Defaults to
                ``AZURE_DOCUMENTINTELLIGENCE_KEY``.
            model_id: The prebuilt model to analyze with. ``"prebuilt-read"`` (the default)
                does OCR/text; ``"prebuilt-layout"`` adds tables/structure.

        Raises:
            MissingExtraError: If the ``azure`` extra is not installed.
        """
        # require_extra is cheap (importlib.util.find_spec) — call it first so selecting an
        # uninstalled backend fails fast with the pip hint, before any other work.
        require_extra(_SLOT, _NAME, _EXTRA, "azure.ai.documentintelligence")
        self._endpoint = endpoint or os.environ.get("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT")
        self._key = key or os.environ.get("AZURE_DOCUMENTINTELLIGENCE_KEY")
        self._model_id = model_id

    def _client(self) -> DocumentIntelligenceClient:
        """Build the vendor Document Intelligence client, lazily importing the SDK.

        Returns:
            A configured vendor ``DocumentIntelligenceClient`` instance. This object is a
            VENDOR type and must never escape this adapter.

        Raises:
            StageError: If the endpoint or key is missing.
        """
        # Lazy: imported only when we actually parse, never at module top level.
        # (Resolved for typing via the TYPE_CHECKING import above; optional extra: azure.)
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import (  # type: ignore[import-not-found]  # optional extra: azure
            AzureKeyCredential,
        )

        if not self._endpoint:
            raise StageError(
                "parse",
                "Azure Document Intelligence requires an endpoint "
                "(set AZURE_DOCUMENTINTELLIGENCE_ENDPOINT or pass endpoint=).",
            )
        if not self._key:
            raise StageError(
                "parse",
                "Azure Document Intelligence requires an API key "
                "(set AZURE_DOCUMENTINTELLIGENCE_KEY or pass key=).",
            )
        return DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._key),
        )

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one file into a :class:`~indx.core.parsed.ParsedDoc` via Document Intelligence.

        Args:
            path: The local source file to upload and analyze.

        Returns:
            The normalized parsed document with extracted text flattened into blocks.

        Raises:
            StageError: If the endpoint/key is missing or the analyze call fails.
        """
        # Build the client BEFORE the try so its clean StageError (missing endpoint/key) is
        # not re-wrapped into the generic ``docintel failed: ...`` message below.
        client = self._client()
        try:
            # SDK 1.0.0 takes the model id positionally and the local bytes via ``body=``
            # (NOT the older ``analyze_request=AnalyzeDocumentRequest(...)`` from the betas).
            # The client is a context manager: closing it per-parse releases the long-lived
            # HTTP transport / connection pool so a corpus of N files doesn't leak N sockets.
            with client, open(path, "rb") as f:
                poller = client.begin_analyze_document(self._model_id, body=f)
                result = poller.result()
        except Exception as exc:  # vendor exception — translate at the edge
            raise StageError("parse", f"docintel failed: {exc}", path=str(path)) from exc

        # Convert at the edge: the vendor result dies here, only ParsedDoc escapes.
        content = result.content or ""
        blocks = [
            Block(
                kind="heading" if raw.startswith("#") else "text",
                text=raw,
                order=order,
            )
            for order, raw in enumerate(_split_paragraphs(content))
        ]
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )
