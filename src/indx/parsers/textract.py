"""TextractParser — cloud parser backend. Requires the ``aws`` extra.

Amazon Textract is a *hosted* OCR service: it uploads the document bytes and authenticates via
boto3's native credential chain, so it is opt-in only and never on the default path
(reference/05-parsers.md). The vendor package (``boto3``) is lazy-imported inside ``__init__`` so
merely importing this module stays cheap and safe on a light install; the ``require_extra`` gate
turns an absent extra into a friendly :class:`~indx.errors.MissingExtraError` at construction time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    import boto3  # type: ignore[import-not-found]  # optional extra: aws

# Sync ``detect_document_text`` caps the inline document at ~10 MB (and a single-page PDF).
# Larger / multi-page inputs need the async S3 flow, which is out of scope for v1.
_MAX_SYNC_BYTES = 10 * 1024 * 1024


class TextractParser:
    """Parser backend that uses Amazon Textract (cloud) to OCR a file into text lines.

    The adapter authenticates through boto3's own credential chain (never hard-coded) and
    flattens Textract's ``LINE`` blocks into :class:`~indx.core.parsed.Block` objects. No vendor
    type is allowed to escape this adapter (reference/05-parsers.md).
    """

    name = "textract"
    version = "1"

    def __init__(self, *, region: str | None = None, profile: str | None = None) -> None:
        """Construct the adapter and verify the optional extra is installed.

        Args:
            region: The AWS region for the Textract client. When ``None`` the adapter falls back
                to ``AWS_REGION`` then ``AWS_DEFAULT_REGION`` from the environment.
            profile: The named AWS profile to use for credentials (a non-secret knob). When
                ``None`` boto3's default credential chain is used.

        Raises:
            MissingExtraError: If the ``aws`` extra is not installed.
        """
        require_extra("parser", "textract", "aws", "boto3")
        self._region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        self._profile = profile

    def _client(self) -> boto3.client:
        """Build the vendor Textract client, lazily importing the SDK.

        Returns:
            A configured vendor ``textract`` client. This object is a VENDOR type and must never
            escape this adapter.
        """
        # Lazy: imported only when we actually parse, never at module top level.
        # (Resolved for typing via the TYPE_CHECKING import above; optional extra: aws.)
        import boto3

        # Credentials come from boto3's own chain (or the named profile); we never read keys.
        session = boto3.Session(profile_name=self._profile) if self._profile else boto3
        return session.client("textract", region_name=self._region)

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one file into a :class:`~indx.core.parsed.ParsedDoc` via Amazon Textract.

        Args:
            path: The local source file to OCR. Must be a single-page document of ≤10 MB.

        Returns:
            The normalized parsed document with each detected ``LINE`` as one text block.

        Raises:
            StageError: If the file exceeds the sync limit or the Textract call fails.
        """
        data = path.read_bytes()
        if len(data) > _MAX_SYNC_BYTES:
            raise StageError(
                "parse",
                f"document is {len(data)} bytes; sync detect_document_text caps at "
                f"{_MAX_SYNC_BYTES} bytes (single-page only) — use the async S3 flow instead",
                path=str(path),
            )

        try:
            resp = self._client().detect_document_text(Document={"Bytes": data})
        except Exception as exc:  # vendor exception — translate at the edge
            raise StageError("parse", f"textract failed: {exc}", path=str(path)) from exc

        # Convert at the edge: vendor response dicts die here, only ParsedDoc escapes.
        lines = [
            b["Text"]
            for b in resp.get("Blocks", [])
            if b.get("BlockType") == "LINE" and "Text" in b
        ]
        blocks = [Block(kind="text", text=line, order=order) for order, line in enumerate(lines)]
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )
