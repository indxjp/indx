"""MarkItDownParser — lightest-install parser backend. Requires the ``markitdown`` extra.

MarkItDown converts a wide range of file types to Markdown with a tiny footprint
(reference/05-parsers.md). This adapter converts once via ``MarkItDown().convert(path)`` and
then flattens the rendered Markdown into :class:`ParsedDoc` ``Block``s using the same
blank-line paragraph split as :class:`~indx.parsers.plaintext.PlainTextParser`, so behavior
stays consistent across the two text-grade parsers.

The vendor result object never leaves this module: the lock-in firewall is enforced at the
adapter edge (coding-standards §6.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions where a NUL byte means "binary masquerading as text" and must be pre-rejected.
# Binary office containers are deliberately absent: they contain NUL legitimately and are
# MarkItDown's job to convert (#20).
_TEXT_LIKE = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".html", ".htm", ".json", ".xml"}
)


class MarkItDownParser:
    """Parser backend that uses Microsoft's MarkItDown to convert files to Markdown.

    The heavy vendor SDK is imported lazily inside :meth:`parse`, and the dependency gate
    fires in :meth:`__init__` so selecting this slot without ``indx[markitdown]`` fails with a
    friendly :class:`~indx.errors.MissingExtraError` at construction time rather than a raw
    ``ImportError`` deep in the vendor package (coding-standards §6.3).
    """

    name = "markitdown"
    version = "1"

    def __init__(self) -> None:
        """Gate construction on the optional ``markitdown`` extra.

        Raises:
            MissingExtraError: If the ``markitdown`` extra is not installed.
        """
        require_extra("parser", "markitdown", "markitdown", "markitdown")

    def parse(self, path: Path) -> ParsedDoc:
        """Convert ``path`` to Markdown and flatten it into a :class:`ParsedDoc`.

        Args:
            path: Local file to parse. This parser never touches the network.

        Returns:
            The normalized :class:`ParsedDoc` with one :class:`Block` per Markdown paragraph,
            tagged ``"heading"`` when the paragraph starts with ``#`` and ``"text"`` otherwise.

        Raises:
            StageError: If MarkItDown fails to convert the file, the input is a NUL-byte
                binary masquerading as text, or the conversion yields no usable text.
        """
        # Binary sniff at the adapter edge, BEFORE convert() — but ONLY for *text-like* inputs.
        # A NUL byte in a file that claims to be text (.md/.txt/.csv/...) means it is binary
        # masquerading as text: MarkItDown does not raise on such input, it returns the literal
        # 4-char string "None" as ``text_content``, which is truthy and would otherwise pass
        # straight through as a chunk and poison retrieval. Reject those here so the parse stage
        # records a skip, exactly as PlainTextParser does (plaintext.py:57 — consistent slots).
        #
        # Binary office containers (.pdf/.docx/.pptx/.xlsx) legitimately contain NUL bytes and are
        # precisely what MarkItDown exists to convert, so the sniff must NOT pre-reject them — that
        # over-broad gate dropped every real office doc to 0 chunks (#20). Let the vendor's own
        # per-format converters handle real binaries; the degenerate-output guard below still
        # rejects anything that yields no usable text, for any extension.
        if path.suffix.lower() in _TEXT_LIKE and b"\x00" in path.read_bytes():
            raise StageError("parse", f"skipped non-text/binary file: {path.name}", path=str(path))

        # Lazy: imported only when we actually parse, never at module top level (§2).
        from markitdown import MarkItDown  # type: ignore[import-not-found]  # optional extra: markitdown  # noqa: E501,I001

        # enable_plugins=False keeps conversion deterministic — no third-party plugins, no
        # nondeterministic ordering or captioning on the default path (§1.4, reference §"Det").
        converter = MarkItDown(enable_plugins=False)
        try:
            result = converter.convert(str(path))
        except Exception as exc:  # vendor exception — translate at the edge (§6.2/§8)
            # MarkItDown raises a MissingDependencyException when the per-format converter for
            # this file type needs an optional dependency that isn't installed (e.g. docx needs
            # ``markitdown[docx]``). It does NOT always surface that as the top-level exception
            # class: docx/pptx failures arrive wrapped as ``FileConversionException`` whose *cause*
            # is the MissingDependencyException, so matching ``type(exc).__name__`` alone missed
            # them. Match the message instead (NO runtime import of the vendor's exception type or
            # any format lib — the firewall keeps vendor objects out) so the hint fires for the
            # wrapped cases too. Otherwise keep the generic message.
            exc_text = f"{type(exc).__name__} {exc}".lower()
            if "missingdependency" in exc_text:
                msg = (
                    f"markitdown is missing an optional dependency for this file type "
                    f"(install the matching format extra, e.g. markitdown[docx] / "
                    f"markitdown[pptx] / markitdown[xlsx]): {exc}"
                )
            else:
                msg = f"markitdown failed: {exc}"
            raise StageError("parse", msg, path=str(path)) from exc

        # `result` is a vendor result object — it must not escape this method.
        # Guard degenerate output: MarkItDown returns ``None`` text_content for inputs it
        # cannot extract (and a bare ``or ""`` would still let an all-whitespace conversion
        # through as a zero-block doc). Treat empty/whitespace-only output as a parse-skip so it
        # flows into the same "indexed with 0 chunks" surface as other skips, never a chunk.
        text = result.text_content or ""
        if not text.strip():
            raise StageError("parse", f"markitdown produced no text: {path.name}", path=str(path))
        blocks: list[Block] = []
        for order, raw in enumerate(_split_paragraphs(text)):
            kind = "heading" if raw.startswith("#") else "text"
            blocks.append(Block(kind=kind, text=raw.strip(), order=order))
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty, blank-line-delimited paragraphs (mirrors PlainTextParser)."""
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]
