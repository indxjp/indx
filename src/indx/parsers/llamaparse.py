"""LlamaParseParser — cloud parser backend. Requires the ``llamaparse`` extra.

LlamaParse is a *hosted* service: it uploads the document and needs an API key, so it is
opt-in only and never on the default path (reference/05-parsers.md). The vendor package is
lazy-imported inside ``__init__`` so merely importing this module stays cheap and safe on a
light install; the ``require_extra`` gate turns an absent extra into a friendly
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
    from llama_cloud_services import (  # type: ignore[import-not-found]  # optional extra: llamaparse
        LlamaParse,
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split rendered markdown into non-empty, stripped paragraphs on blank lines.

    Mirrors :class:`~indx.parsers.plaintext.PlainTextParser` so block segmentation stays
    consistent across markdown-producing parsers.

    Args:
        text: The full markdown text returned by LlamaParse.

    Returns:
        The ordered list of non-empty paragraph strings.
    """
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]


class LlamaParseParser:
    """Parser backend that uses LlamaParse (cloud) to convert a file to markdown.

    The adapter reads ``LLAMA_CLOUD_API_KEY`` from the environment (never hard-coded) and
    flattens LlamaParse's markdown output into :class:`~indx.core.parsed.Block` objects.
    No vendor type is allowed to escape this adapter (reference/05-parsers.md).
    """

    name = "llamaparse"
    version = "1"

    def __init__(self, *, result_type: str = "markdown") -> None:
        """Construct the adapter and verify the optional extra is installed.

        Args:
            result_type: The LlamaParse output mode. ``"markdown"`` (the default) keeps
                downstream block segmentation stable; ``"text"`` is also accepted.

        Raises:
            MissingExtraError: If the ``llamaparse`` extra is not installed.
        """
        require_extra("parser", "llamaparse", "llamaparse", "llama_cloud_services")
        self._result_type = result_type
        self._api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

    def _parser(self) -> LlamaParse:
        """Build the vendor LlamaParse client, lazily importing the SDK.

        Returns:
            A configured vendor ``LlamaParse`` instance. This object is a VENDOR type and
            must never escape this adapter.
        """
        # Lazy: imported only when we actually parse, never at module top level.
        # (Resolved for typing via the TYPE_CHECKING import above; optional extra: llamaparse.)
        from llama_cloud_services import LlamaParse

        return LlamaParse(api_key=self._api_key, result_type=self._result_type)

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one file into a :class:`~indx.core.parsed.ParsedDoc` via LlamaParse.

        Args:
            path: The local source file to upload and parse.

        Returns:
            The normalized parsed document with markdown flattened into blocks.

        Raises:
            StageError: If the LlamaParse call fails for any reason.
        """
        try:
            documents = self._parser().load_data(str(path))
        except Exception as exc:  # vendor exception — translate at the edge
            raise StageError("parse", f"llamaparse failed: {exc}", path=str(path)) from exc

        # Convert at the edge: vendor Document objects die here, only ParsedDoc escapes.
        markdown = "\n\n".join(doc.text for doc in documents)
        blocks = [
            Block(
                kind="heading" if raw.startswith("#") else "text",
                text=raw,
                order=order,
            )
            for order, raw in enumerate(_split_paragraphs(markdown))
        ]
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )
