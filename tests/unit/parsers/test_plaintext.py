"""Unit tests for :class:`~indx.parsers.plaintext.PlainTextParser`.

The plaintext parser is the zero-dependency core fallback used by the offline corpus suite.
These tests pin its binary-sniffing contract (bug #7): binary / non-UTF-8 input must *raise*
(so the parse stage records the file as a skip) while legitimate non-ASCII UTF-8 text — accents,
Japanese, emoji — still round-trips into clean blocks with no U+FFFD replacement chars.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.parsers.plaintext import PlainTextParser


def test_nul_byte_file_raises(tmp_path: Path) -> None:
    # A NUL byte is the canonical binary tell; force-decoding would embed \x00 mojibake.
    src = tmp_path / "blob.bin"
    src.write_bytes(b"hello\x00world")
    with pytest.raises(Exception):  # noqa: B017 - any exception becomes a recorded skip
        PlainTextParser().parse(src)


def test_invalid_utf8_file_raises(tmp_path: Path) -> None:
    src = tmp_path / "random.bin"
    src.write_bytes(b"\xff\xfe\x00\x01" * 100)
    with pytest.raises(Exception):  # noqa: B017
        PlainTextParser().parse(src)


def test_non_ascii_utf8_round_trips_without_replacement_chars(tmp_path: Path) -> None:
    # Accented Latin-1-via-UTF-8, Japanese, and an emoji must all survive intact.
    src = tmp_path / "doc.md"
    src.write_text("# Café éà\n\nこんにちは\n\nDone ✅\n", encoding="utf-8")

    parsed = PlainTextParser().parse(src)

    texts = [b.text for b in parsed.blocks]
    assert texts == ["# Café éà", "こんにちは", "Done ✅"]
    assert all("�" not in b.text for b in parsed.blocks)


def test_heading_detection_still_works(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nBody paragraph.\n", encoding="utf-8")

    parsed = PlainTextParser().parse(src)

    kinds = [b.kind for b in parsed.blocks]
    assert kinds == ["heading", "text"]
