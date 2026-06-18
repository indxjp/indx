"""PlainTextParser — zero-dependency fallback parser that ships in core.

Splits text files on blank lines into blocks. Enough to run the whole pipeline air-gapped
with no extras (file-architecture §5), and the parser used by the offline corpus suite.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from indx.core.parsed import Block, ParsedDoc

# Elements whose content is markup/data, not human-readable prose: their text must be dropped.
_NON_CONTENT_TAGS = frozenset({"script", "style", "head", "title", "noscript", "template"})
# Block-level tags that imply a paragraph/line boundary when stripping tags to plain text.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "tr",
        "table",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "aside",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)


class PlainTextParser:
    name = "plaintext"
    version = "1"

    def parse(self, path: Path) -> ParsedDoc:
        # Decode with a fixed, deterministic, content-keyed ladder so legacy-encoded text (Latin-1
        # / CP1252) and BOM'd UTF-16 survive instead of being dropped as "binary", while true
        # binaries are still rejected (so the parse stage records them as a skip). The order is
        # fixed — never charset-sniffed — so the same bytes decode to the same text on every
        # machine.
        data = Path(path).read_bytes()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            # A UTF-16 byte-order mark unambiguously means UTF-16 text. This MUST come before the
            # NUL-byte binary sniff below: UTF-16 encodes every ASCII/Latin character with a zero
            # byte (``\n`` -> ``0A 00``), so the sniff would otherwise misclassify all real UTF-16
            # text as binary. The BOM fixes endianness, so the decode is deterministic; only a
            # genuinely malformed UTF-16 stream (odd length / lone surrogate) is rejected.
            try:
                text = data.decode("utf-16")
            except UnicodeDecodeError as exc:
                raise ValueError(f"skipped non-text/binary file: {path.name}") from exc
        else:
            # No UTF-16 BOM: a NUL byte now reliably marks a true binary (force-decoding would
            # embed ``\x00`` mojibake that poisons retrieval), so reject it.
            if b"\x00" in data:
                raise ValueError(f"skipped non-text/binary file: {path.name}")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                # CP1252 (a Latin-1 superset) is the catch-all for legacy 8-bit text. Five byte
                # positions (0x81/0x8D/0x8F/0x90/0x9D) are unmapped, so decode with
                # ``errors="replace"`` to guarantee legacy text is never dropped — deterministic,
                # and far better than silently losing the whole file.
                text = data.decode("cp1252", errors="replace")
        # L4: HTML files would otherwise be indexed as raw tag soup ('<!DOCTYPE html>',
        # 'href', …). Detect HTML by extension or a leading doctype/<html> sniff and strip to
        # readable visible text (stdlib only — stays offline & deterministic) BEFORE the same
        # paragraph split below. Non-HTML input is left byte-for-byte unchanged.
        if _is_html(path, text):
            text = _html_to_text(text)
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
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in paras if p]


def _is_html(path: Path, text: str) -> bool:
    """True if ``path``/``text`` looks like HTML: by extension or a leading doctype/<html> tag."""
    if path.suffix.lower() in {".html", ".htm"}:
        return True
    # Content sniff: skip a leading BOM/whitespace, then look for the doctype or an <html> tag.
    head = text.lstrip("﻿ \t\r\n").lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, dropping script/style/head content.

    Block-level tags insert blank-line separators so the downstream paragraph split produces
    sensible paragraphs. Output is post-processed (whitespace collapse) by the caller.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _NON_CONTENT_TAGS:
            self._suppress_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        # Self-closing block tags (e.g. <br/>) still imply a boundary.
        if tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _NON_CONTENT_TAGS and self._suppress_depth > 0:
            self._suppress_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._suppress_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    """Strip HTML to readable text: drop script/style, unescape entities, collapse whitespace."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    text = unescape(extractor.get_text())
    # Collapse runs of spaces/tabs within lines, normalize paragraph breaks to a blank line.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Re-join non-empty lines, preserving blank-line paragraph boundaries.
    out: list[str] = []
    prev_blank = True  # leading blanks are dropped
    for line in lines:
        if line:
            out.append(line)
            prev_blank = False
        elif not prev_blank:
            out.append("")
            prev_blank = True
    return "\n".join(out)
