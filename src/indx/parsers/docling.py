"""DoclingParser — the default parser. Requires the ``docling`` extra.

Docling is a local, high-fidelity layout/table/figure parser. This adapter converts one
source file into indx's normalized :class:`~indx.core.parsed.ParsedDoc` and never lets a
Docling vendor type escape the :meth:`DoclingParser.parse` boundary (coding-standards §6.2).

The heavy ``docling`` dependency is imported lazily inside :meth:`__init__`/:meth:`parse`,
so merely importing this module is always safe on a light core install (coding-standards §6.3).
Selecting this parser without the extra installed raises
:class:`~indx.errors.MissingExtraError` at construction time.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from indx.core.parsed import Block, ParsedDoc
from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from docling.document_converter import (  # type: ignore[import-not-found]  # optional extra: docling
        DocumentConverter,
    )

logger = logging.getLogger(__name__)

# Map Docling item ``label`` values onto the small ParsedDoc kind vocabulary
# (text | heading | table | caption | code | ...). Anything unmapped falls back to "text".
_LABEL_TO_KIND: dict[str, str] = {
    "title": "heading",
    "section_header": "heading",
    "page_header": "heading",
    "subtitle": "heading",
    "caption": "caption",
    "code": "code",
    "formula": "code",
    "table": "table",
    "footnote": "text",
    "paragraph": "text",
    "text": "text",
    "list_item": "text",
}

# Conversion-status names Docling reports for a usable result. An unsupported input format
# (e.g. ``.odt``/``.rst``) does NOT raise — Docling returns a result whose ``status`` is
# ``SKIPPED``/``FAILURE`` and whose document is empty. We compare by status *name* rather than
# importing the vendor enum so the check stays a pure boundary guard (no vendor type leaks, §6.2)
# and degrades safely on Docling shapes/fakes that expose no ``status`` at all.
_OK_STATUS = frozenset({"SUCCESS", "PARTIAL_SUCCESS"})


class DoclingParser:
    """Default parser backend built on Docling's :class:`DocumentConverter`.

    Converts a local file into a :class:`~indx.core.parsed.ParsedDoc` by walking the native
    ``DoclingDocument`` in reading order and flattening each item into a
    :class:`~indx.core.parsed.Block`. Vendor types are confined to this module.

    Attributes:
        name: Registry key for this backend (``"docling"``).
        version: Adapter version, recorded on every emitted ``ParsedDoc``.
    """

    name = "docling"
    version = "1"

    def __init__(self) -> None:
        """Construct the parser, failing fast if the ``docling`` extra is absent.

        Raises:
            MissingExtraError: If ``indx[docling]`` is not installed.
        """
        # Import-safety (§6.3): the gate lives here, not at module top level, so importing
        # this module never crashes a light core install.
        require_extra("parser", "docling", "docling", "docling")
        self._converter: DocumentConverter | None = None
        # Guards both the one-time converter build (construction race) and concurrent
        # convert() calls, since DocumentConverter is not verified thread-safe and the
        # default pipeline fans parse() across a ThreadPoolExecutor.
        self._lock = threading.Lock()

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one local file into a normalized :class:`ParsedDoc`.

        Args:
            path: Local filesystem path to the source document. Docling parses local
                paths only; this adapter never touches the network (constraint 1).

        Returns:
            A :class:`ParsedDoc` whose blocks are the document's items in reading order.

        Raises:
            StageError: If Docling fails to convert the file.
        """
        converter = self._get_converter()
        try:
            # Serialize convert() under the same lock: DocumentConverter is not verified
            # safe for concurrent convert() on one instance under the default thread pool.
            with self._lock:
                result = converter.convert(str(path))
        except Exception as exc:  # vendor exception — translate at the edge (§8)
            raise StageError("parse", f"docling failed: {exc}", path=str(path)) from exc

        # A non-success status is Docling's *non-raising* failure channel: an unsupported format
        # yields a SKIPPED/FAILURE result with an empty document instead of an exception. Left
        # unchecked it would flow through as an empty ParsedDoc — 0 chunks, silently unindexed,
        # with parse_failures=0 (issue #17 bug 1). Surface it as a StageError so ResumableParseStage
        # records a visible per-file skip (and --strict promotes it to fatal).
        _raise_if_not_converted(result, path)

        # ``document`` is a DoclingDocument — a VENDOR type that must die inside this method.
        document = result.document
        blocks = self._to_blocks(document)
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=blocks,
        )

    def _get_converter(self) -> DocumentConverter:
        """Return a cached :class:`DocumentConverter`, building it once per instance.

        Returns:
            The lazily-constructed Docling converter.
        """
        # Double-checked locking: avoid N threads each building a heavyweight converter
        # (and a nondeterministic last-writer instance) on the default parse path.
        if self._converter is None:
            with self._lock:
                if self._converter is None:
                    # Lazy: imported only when we actually parse, never at module top level.
                    # (Resolved for typing via the TYPE_CHECKING import above; optional
                    # extra: docling.)
                    from docling.document_converter import DocumentConverter

                    self._converter = DocumentConverter()
        return self._converter

    def _to_blocks(self, document: object) -> list[Block]:
        """Flatten a ``DoclingDocument`` into ordered :class:`Block` objects.

        Iterates the document's items in Docling's deterministic reading order so
        ``Block.order`` is stable run-to-run (constraint 4). Empty items are skipped.

        Args:
            document: The native ``DoclingDocument`` (vendor type, never leaks out).

        Returns:
            Blocks in document reading order, each with a stable ``order`` index.
        """
        blocks: list[Block] = []
        order = 0
        for item in _iter_items(document):
            text = _text_of(item)
            if not text:
                continue
            blocks.append(Block(kind=_kind_of(item), text=text, order=order))
            order += 1
        return blocks


def _raise_if_not_converted(result: object, path: Path) -> None:
    """Raise :class:`StageError` when Docling reports a non-success conversion ``status``.

    Docling signals an unsupported/unreadable input by returning a result with a non-success
    ``status`` (and an empty document) rather than raising. We translate that into the same
    per-file skip a raised vendor error would produce. A result that exposes no ``status`` (older
    Docling shapes / test fakes) is treated as success — this guard only ever adds failures that
    Docling itself flagged.

    Args:
        result: The native ``ConversionResult`` (vendor type, never leaks out of this module).
        path: The source path, for the error message.

    Raises:
        StageError: If ``result.status`` is present and not one of :data:`_OK_STATUS`.
    """
    status = getattr(result, "status", None)
    if status is None:
        return
    status_name = str(getattr(status, "name", status)).upper()
    if status_name in _OK_STATUS:
        return
    errors = getattr(result, "errors", None) or []
    reason = "; ".join(str(getattr(e, "error_message", e)) for e in errors) or status_name
    raise StageError(
        "parse", f"docling could not convert ({status_name}): {reason}", path=str(path)
    )


def _iter_items(document: object) -> list[object]:
    """Yield a document's content items in a deterministic reading order.

    Prefers Docling's ``iterate_items()`` (reading-order traversal); falls back to the
    flat ``texts`` collection if that method is unavailable.

    Args:
        document: A ``DoclingDocument``.

    Returns:
        The native items in a fixed order.
    """
    iterate = getattr(document, "iterate_items", None)
    if callable(iterate):
        items: list[object] = []
        for entry in iterate():
            # ``iterate_items`` yields ``(item, level)`` tuples; older shapes yield items.
            if isinstance(entry, tuple) and entry:
                items.append(entry[0])
            else:
                items.append(entry)
        return items
    return list(getattr(document, "texts", []))


def _kind_of(item: object) -> str:
    """Map a native item's ``label`` to a ParsedDoc block kind.

    Args:
        item: A Docling item exposing an optional ``label``.

    Returns:
        One of the ParsedDoc kind strings; defaults to ``"text"``.
    """
    label = getattr(item, "label", None)
    label_str = str(getattr(label, "value", label) or "").lower()
    return _LABEL_TO_KIND.get(label_str, "text")


def _text_of(item: object) -> str:
    """Extract the text payload of a native item.

    Handles plain text items (``.text``) and tables (``export_to_markdown()``).

    Args:
        item: A Docling item.

    Returns:
        The item's stripped text, or an empty string if it carries none.
    """
    export_md = getattr(item, "export_to_markdown", None)
    if callable(export_md) and getattr(item, "text", None) is None:
        try:
            return str(export_md()).strip()
        except Exception:  # noqa: BLE001 - tables without a doc context can't render
            return ""
    text = getattr(item, "text", None)
    return str(text).strip() if text is not None else ""
