"""UnstructuredParser — broad-format parser backend. Requires the `unstructured` extra.

Wraps :func:`unstructured.partition.auto.partition`, which auto-detects a file's type and
returns a list of native ``Element`` objects. This adapter converts each element at the edge
into a :class:`~indx.core.parsed.Block` and never lets a vendor type escape :meth:`parse`
(file-architecture §5, parsers reference §"convert at the edge").

Unstructured's *core* install covers plain text, HTML, XML, JSON, and email with no extra
dependencies. Richer formats (``.docx``, ``.pptx``, PDFs, images) require Unstructured's own
sub-extras — e.g. ``pip install 'unstructured[docx,pptx]'`` or ``unstructured[all-docs]``.
indx exposes only the base ``indx[unstructured]`` extra and never silently pulls ``all-docs``;
install the sub-extra your corpus needs alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from pathlib import Path

# Unstructured element categories (the ``.category`` string on each Element) mapped onto the
# small open ``Block.kind`` vocabulary (text | heading | table | caption | code | list).
# Any category not listed here falls back to "text" via ``dict.get``.
_CATEGORY_TO_KIND: dict[str, str] = {
    "Title": "heading",
    "Header": "heading",
    "SectionHeader": "heading",
    "Table": "table",
    "FigureCaption": "caption",
    "Caption": "caption",
    "CodeSnippet": "code",
    "ListItem": "list",
    "ListItemOther": "list",
}


class UnstructuredParser:
    """Parser backed by Unstructured's ``partition`` auto-detection.

    Attributes:
        name: Registry key used to select this parser (``"unstructured"``).
        version: Adapter version recorded on every produced :class:`ParsedDoc`.
    """

    name = "unstructured"
    version = "1"

    def __init__(self) -> None:
        """Verify the optional extra is installed before the parser can be used.

        Calling :func:`require_extra` here (rather than at module import time) keeps importing
        this module cheap and side-effect free, while still failing fast — with an actionable
        ``pip install indx[unstructured]`` hint — the moment the slot is constructed without
        the extra present (file-architecture §5).

        Raises:
            MissingExtraError: If the ``unstructured`` package is not importable.
        """
        require_extra("parser", "unstructured", "unstructured", "unstructured")

    def parse(self, path: Path) -> ParsedDoc:
        """Partition one file into a normalized :class:`ParsedDoc`.

        Args:
            path: Local path to the source file. Unstructured auto-detects its type. This
                parser is local-only and must not touch the network.

        Returns:
            A :class:`ParsedDoc` whose blocks mirror the partitioned elements in their original
            (deterministic) order, with categories mapped to ``Block.kind``.

        Raises:
            StageError: If Unstructured fails to partition the file. The vendor exception is
                translated at the edge so nothing vendor-specific escapes the adapter.
        """
        # Lazy: imported only when we actually parse, never at module top level, so a light
        # install can import this module without the heavy dependency present.
        from unstructured.partition.auto import (  # type: ignore[import-not-found]  # optional extra: unstructured
            partition,
        )

        try:
            elements = partition(filename=str(path))  # list[Element] — vendor type, stays local
        except Exception as exc:  # noqa: BLE001 - translate any vendor failure at the edge
            raise StageError("parse", f"unstructured failed: {exc}", path=str(path)) from exc

        blocks = [
            Block(
                kind=_category_to_kind(getattr(element, "category", None)),
                text=getattr(element, "text", "") or "",
                order=order,
            )
            for order, element in enumerate(elements)
        ]
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )


def _category_to_kind(category: str | None) -> str:
    """Map an Unstructured element category onto a :class:`Block` kind.

    Args:
        category: The element's ``.category`` string, or ``None`` if absent.

    Returns:
        The corresponding ``Block.kind``; unknown or missing categories map to ``"text"``.
    """
    if category is None:
        return "text"
    return _CATEGORY_TO_KIND.get(category, "text")
