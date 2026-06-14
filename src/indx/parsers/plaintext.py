"""PlainTextParser — zero-dependency fallback parser that ships in core.

Splits text files on blank lines into blocks. Enough to run the whole pipeline air-gapped
with no extras (file-architecture §5), and the parser used by the offline corpus suite.
"""

from __future__ import annotations

from pathlib import Path

from indx.core.parsed import Block, ParsedDoc


class PlainTextParser:
    name = "plaintext"
    version = "1"

    def parse(self, path: Path) -> ParsedDoc:
        # Sniff for binary/non-text content before the tolerant read_text: a NUL byte or a
        # strict-UTF-8 decode failure means force-decoding would produce U+FFFD/\x00 mojibake
        # that poisons retrieval. Raise so the parse stage records this file as a skip rather
        # than emitting garbage blocks (read_text stays tolerant for the messy_real corpus).
        data = Path(path).read_bytes()
        if b"\x00" in data:
            raise ValueError(f"skipped non-text/binary file: {path.name}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"skipped non-text/binary file: {path.name}") from exc
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
