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


def test_html_input_yields_tag_free_text(tmp_path: Path) -> None:
    # L4: HTML through the offline plaintext parser must be stripped to readable text, not
    # indexed as raw tag soup. No '<', 'href', 'DOCTYPE', or script/style content may survive.
    src = tmp_path / "python-faq.html"
    src.write_text(
        "<!DOCTYPE html>\n"
        "<html><head><title>Python FAQ</title>\n"
        "<style>body { color: red; }</style>\n"
        "<script>var href = 'tracking';</script></head>\n"
        "<body>\n"
        "<h1>Python FAQ</h1>\n"
        '<p>See the <a href="https://example.com">docs</a> for &amp; details.</p>\n'
        "<p>Second paragraph here.</p>\n"
        "</body></html>\n",
        encoding="utf-8",
    )

    parsed = PlainTextParser().parse(src)
    full_text = "\n".join(b.text for b in parsed.blocks)

    # No raw markup leaks through.
    assert "<" not in full_text
    assert ">" not in full_text
    assert "href" not in full_text
    assert "DOCTYPE" not in full_text
    # script/style payloads are dropped entirely.
    assert "color: red" not in full_text
    assert "tracking" not in full_text
    assert "var " not in full_text
    # Visible prose survives, entities are unescaped, link text is preserved inline.
    assert "Python FAQ" in full_text
    assert "See the docs for & details." in full_text
    assert "Second paragraph here." in full_text


def test_html_detected_by_content_sniff_without_extension(tmp_path: Path) -> None:
    # No .html extension: a leading <html>/doctype must still trigger tag stripping.
    src = tmp_path / "page.txt.bak"
    src.write_text("<html><body><p>Hello world</p></body></html>", encoding="utf-8")

    parsed = PlainTextParser().parse(src)
    texts = [b.text for b in parsed.blocks]

    assert texts == ["Hello world"]


def test_non_html_text_unchanged(tmp_path: Path) -> None:
    # A non-HTML file that merely contains angle brackets must NOT be treated as HTML, so the
    # existing plaintext behavior (and the offline golden corpus) is preserved exactly.
    src = tmp_path / "note.txt"
    src.write_text("Use a < b and c > d in code.\n\nSecond para.", encoding="utf-8")

    parsed = PlainTextParser().parse(src)
    texts = [b.text for b in parsed.blocks]

    assert texts == ["Use a < b and c > d in code.", "Second para."]
