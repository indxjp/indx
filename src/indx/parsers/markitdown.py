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
            StageError: If MarkItDown fails to convert the file.
        """
        # Lazy: imported only when we actually parse, never at module top level (§2).
        from markitdown import MarkItDown  # type: ignore[import-not-found]  # optional extra: markitdown  # noqa: E501,I001

        # enable_plugins=False keeps conversion deterministic — no third-party plugins, no
        # nondeterministic ordering or captioning on the default path (§1.4, reference §"Det").
        converter = MarkItDown(enable_plugins=False)
        try:
            result = converter.convert(str(path))
        except Exception as exc:  # vendor exception — translate at the edge (§6.2/§8)
            # MarkItDown raises a MissingDependencyException when the per-format converter
            # for this file type needs an optional dependency that isn't installed (e.g. its
            # PDF support, pulled by ``markitdown[pdf]``). Detect that flavor by the vendor
            # exception's class name (NO runtime import of the vendor's exception type or any
            # pdf lib — the firewall keeps vendor objects out) and surface a clear hint so a
            # parse skip isn't mistaken for a corrupt file. Otherwise keep the generic message.
            if "missingdependency" in type(exc).__name__.lower():
                msg = (
                    f"markitdown is missing an optional dependency for this file type "
                    f"(likely a format whose extra is not installed, e.g. PDF needs "
                    f"markitdown[pdf]): {exc}"
                )
            else:
                msg = f"markitdown failed: {exc}"
            raise StageError("parse", msg, path=str(path)) from exc

        # `result` is a vendor result object — it must not escape this method.
        text = result.text_content or ""
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
