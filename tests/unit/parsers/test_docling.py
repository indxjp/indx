"""Unit tests for :class:`indx.parsers.docling.DoclingParser`.

The ``docling`` extra is not installed in CI, so the vendor SDK is faked and injected into
``sys.modules``. Tests are fully offline and deterministic (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from indx.errors import MissingExtraError
from indx.parsers.base import Parser
from indx.parsers.docling import DoclingParser

if TYPE_CHECKING:
    from collections.abc import Iterator


# --- Fake Docling SDK -------------------------------------------------------------------


class _FakeLabel:
    """Mimics Docling's enum-like label exposing a ``.value``."""

    def __init__(self, value: str) -> None:
        self.value = value


class _FakeTextItem:
    """A Docling text item: carries ``text`` and a ``label``."""

    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label = _FakeLabel(label)


class _FakeTableItem:
    """A Docling table item: no ``text``, renders via ``export_to_markdown``."""

    def __init__(self, markdown: str) -> None:
        self.text = None
        self.label = _FakeLabel("table")
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeDocument:
    """A fake ``DoclingDocument`` exposing ``iterate_items`` in reading order."""

    def __init__(self, items: list[object]) -> None:
        self._items = items

    def iterate_items(self) -> Iterator[tuple[object, int]]:
        for item in self._items:
            yield item, 0


class _FakeResult:
    def __init__(self, document: _FakeDocument) -> None:
        self.document = document


class _FakeConverter:
    """Stand-in for ``docling.document_converter.DocumentConverter``."""

    last_source: str | None = None

    def convert(self, source: str) -> _FakeResult:
        _FakeConverter.last_source = source
        document = _FakeDocument(
            [
                _FakeTextItem("Big Title", "title"),
                _FakeTextItem("A normal paragraph.", "text"),
                _FakeTextItem("", "text"),  # empty -> skipped
                _FakeTableItem("| a | b |\n| - | - |\n| 1 | 2 |"),
                _FakeTextItem("Section Two", "section_header"),
            ]
        )
        return _FakeResult(document)


def _install_fake_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the extra gate and inject a fake ``docling`` package."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # docling.py binds the name at import time, so patch its module namespace too.
    monkeypatch.setattr("indx.parsers.docling.require_extra", lambda *a, **k: None)

    docling_pkg = types.ModuleType("docling")
    converter_mod = types.ModuleType("docling.document_converter")
    converter_mod.DocumentConverter = _FakeConverter  # type: ignore[attr-defined]
    docling_pkg.document_converter = converter_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "docling", docling_pkg)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_mod)


# --- Tests ------------------------------------------------------------------------------


def test_satisfies_parser_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_docling(monkeypatch)
    parser = DoclingParser()
    assert isinstance(parser, Parser)
    assert parser.name == "docling"


def test_parse_round_trips_to_parsed_doc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_docling(monkeypatch)
    src = tmp_path / "doc.pdf"
    src.write_text("ignored by fake converter")

    parser = DoclingParser()
    parsed = parser.parse(src)

    assert parsed.parser == "docling"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)
    assert _FakeConverter.last_source == str(src)

    # Empty item dropped; order is contiguous and reflects reading order.
    assert [b.order for b in parsed.blocks] == [0, 1, 2, 3]
    assert [b.kind for b in parsed.blocks] == ["heading", "text", "table", "heading"]
    assert parsed.blocks[0].text == "Big Title"
    assert parsed.blocks[2].kind == "table"
    assert "| a | b |" in parsed.blocks[2].text


def test_parse_translates_vendor_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_docling(monkeypatch)

    class _Boom(_FakeConverter):
        def convert(self, source: str) -> _FakeResult:
            raise RuntimeError("kaboom")

    monkeypatch.setattr(
        sys.modules["docling.document_converter"],
        "DocumentConverter",
        _Boom,
    )

    parser = DoclingParser()
    with pytest.raises(Exception) as excinfo:
        parser.parse(tmp_path / "bad.pdf")
    # StageError names the stage and the offending path.
    assert "parse" in str(excinfo.value)
    assert "kaboom" in str(excinfo.value)


def test_construction_without_extra_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do NOT neutralize require_extra: in the real dep-absent env this must raise.
    monkeypatch.delitem(sys.modules, "docling", raising=False)
    with pytest.raises(MissingExtraError) as excinfo:
        DoclingParser()
    assert "pip install indx[docling]" in str(excinfo.value)
