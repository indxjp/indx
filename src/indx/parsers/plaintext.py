"""PlainTextParser — zero-dependency fallback parser that ships in core.

Splits text files on blank lines into blocks. Enough to run the whole pipeline air-gapped
with no extras (file-architecture §5), and the parser used by the offline corpus suite.
"""

from __future__ import annotations

from pathlib import Path

from indx.core.parsed import Block, ParsedDoc
from indx.utils.io import read_text


class PlainTextParser:
    name = "plaintext"
    version = "1"

    def parse(self, path: Path) -> ParsedDoc:
        text = read_text(path)
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
