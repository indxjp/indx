"""Unit tests for :class:`~indx.parsers.markitdown.MarkItDownParser`.

The ``markitdown`` extra is absent in CI, so the vendor SDK is faked: a stub ``markitdown``
module is injected into ``sys.modules`` and the dependency gate is neutralized. The tests then
prove the adapter satisfies the :class:`~indx.parsers.base.Parser` protocol and round-trips the
fake conversion into a valid :class:`~indx.core.parsed.ParsedDoc`. One test runs against the
real (dep-absent) environment and asserts construction raises
:class:`~indx.errors.MissingExtraError`. All tests are deterministic and offline.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from indx.errors import MissingExtraError
from indx.parsers.base import Parser
from indx.parsers.markitdown import MarkItDownParser

if TYPE_CHECKING:
    from collections.abc import Iterator

# Markdown the fake converter returns: a heading paragraph and two text paragraphs.
_FAKE_MARKDOWN = "# Title\n\nAlpha beta gamma.\n\nDelta epsilon zeta.\n"


class _FakeResult:
    """Mimics the MarkItDown result object: exposes ``text_content``."""

    def __init__(self, text: str) -> None:
        self.text_content = text


class _FakeMarkItDown:
    """Mimics ``markitdown.MarkItDown`` for offline tests."""

    def __init__(self, *, enable_plugins: bool = False) -> None:
        # Determinism guard: the adapter must keep plugins off.
        assert enable_plugins is False
        self.enable_plugins = enable_plugins

    def convert(self, source: str) -> _FakeResult:
        return _FakeResult(_FAKE_MARKDOWN)


@pytest.fixture
def fake_markitdown(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject a fake ``markitdown`` module and neutralize the dependency gate.

    The adapter binds ``require_extra`` at import time (``from ... import require_extra``), so
    the patch must target that bound name in the adapter module, not ``indx.utils.lazy``.
    """
    monkeypatch.setattr("indx.parsers.markitdown.require_extra", lambda *a, **k: None)
    fake_module = types.ModuleType("markitdown")
    fake_module.MarkItDown = _FakeMarkItDown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)
    yield


def test_satisfies_parser_protocol(fake_markitdown: None) -> None:
    assert isinstance(MarkItDownParser(), Parser)


def test_parse_round_trips_into_parsed_doc(fake_markitdown: None, tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("ignored — the fake converter supplies the markdown")

    parsed = MarkItDownParser().parse(src)

    assert parsed.parser == "markitdown"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)

    kinds = [b.kind for b in parsed.blocks]
    texts = [b.text for b in parsed.blocks]
    orders = [b.order for b in parsed.blocks]
    assert kinds == ["heading", "text", "text"]
    assert texts == ["# Title", "Alpha beta gamma.", "Delta epsilon zeta."]
    assert orders == [0, 1, 2]


def test_parse_skips_nul_byte_binary_no_none_chunk(fake_markitdown: None, tmp_path: Path) -> None:
    # #14 N1: MarkItDown returns the literal string "None" for a NUL-byte file masquerading as
    # text; the adapter must reject it BEFORE convert() (binary sniff), so no "None" chunk is ever
    # produced — matching the plaintext parser's rejection of the same file.
    from indx.errors import StageError

    src = tmp_path / "fake-binary.md"
    src.write_bytes(b"PK\x03\x04\x00\x00fake binary content\x00\x00\xff\xfe")
    with pytest.raises(StageError) as excinfo:
        MarkItDownParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


@pytest.mark.parametrize("ext", [".docx", ".pdf", ".pptx", ".xlsx"])
def test_parse_does_not_pre_reject_binary_extension(
    fake_markitdown: None, tmp_path: Path, ext: str
) -> None:
    # #20 Bug 1: binary office containers legitimately contain NUL bytes and are exactly what
    # MarkItDown converts. The pre-gate must NOT reject them on the NUL sniff — they must reach
    # convert() (here the fake) and round-trip into a ParsedDoc. This locks the over-broad gate
    # (which dropped every real office doc to 0 chunks) from being reintroduced.
    src = tmp_path / f"office{ext}"
    src.write_bytes(b"PK\x03\x04\x00\x00real office container\x00\x00\xff\xfe")

    parsed = MarkItDownParser().parse(src)

    assert parsed.parser == "markitdown"
    assert [b.text for b in parsed.blocks] == [
        "# Title",
        "Alpha beta gamma.",
        "Delta epsilon zeta.",
    ]


def test_parse_skips_empty_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # #14 N1: when MarkItDown yields None / whitespace-only text_content, the adapter must treat it
    # as a parse-skip rather than stringifying it into a chunk.
    from indx.errors import StageError

    monkeypatch.setattr("indx.parsers.markitdown.require_extra", lambda *a, **k: None)

    class _EmptyMarkItDown:
        def __init__(self, *, enable_plugins: bool = False) -> None:
            pass

        def convert(self, source: str) -> _FakeResult:
            return _FakeResult(None)  # type: ignore[arg-type]  # vendor can return None text

    fake_module = types.ModuleType("markitdown")
    fake_module.MarkItDown = _EmptyMarkItDown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    src = tmp_path / "doc.md"
    src.write_text("text without a NUL byte, but the converter yields no usable text")
    with pytest.raises(StageError) as excinfo:
        MarkItDownParser().parse(src)
    assert excinfo.value.stage == "parse"


def test_parse_translates_vendor_failure_to_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from indx.errors import StageError

    monkeypatch.setattr("indx.parsers.markitdown.require_extra", lambda *a, **k: None)

    class _BoomMarkItDown:
        def __init__(self, *, enable_plugins: bool = False) -> None:
            pass

        def convert(self, source: str) -> _FakeResult:
            raise RuntimeError("vendor blew up")

    fake_module = types.ModuleType("markitdown")
    fake_module.MarkItDown = _BoomMarkItDown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    src = tmp_path / "doc.md"
    src.write_text("x")
    with pytest.raises(StageError) as excinfo:
        MarkItDownParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_parse_hint_fires_for_wrapped_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # #20 Bug 3: markitdown wraps a docx/pptx missing-extra failure as FileConversionException
    # whose *message* references the MissingDependencyException cause; the top-level class name is
    # NOT "MissingDependencyException". The adapter must still emit the install-the-extra hint,
    # so it matches on the message text, not just the exception class name.
    from indx.errors import StageError

    monkeypatch.setattr("indx.parsers.markitdown.require_extra", lambda *a, **k: None)

    class _WrappedMarkItDown:
        def __init__(self, *, enable_plugins: bool = False) -> None:
            pass

        def convert(self, source: str) -> _FakeResult:
            raise RuntimeError(  # stands in for FileConversionException
                "File conversion failed after 1 attempt: MissingDependencyException - "
                "DocxConverter recognized the input as .docx but needs markitdown[docx]"
            )

    fake_module = types.ModuleType("markitdown")
    fake_module.MarkItDown = _WrappedMarkItDown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    src = tmp_path / "report.docx"
    src.write_bytes(b"PK\x03\x04\x00\x00")
    with pytest.raises(StageError) as excinfo:
        MarkItDownParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert "missing an optional dependency" in str(excinfo.value)
    assert "markitdown[docx]" in str(excinfo.value)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real dep-absent environment the gate must fire. Guard with
    # find_spec (what require_extra itself uses) so the test skips rather than fails if a
    # contributor has the optional extra installed locally.
    if importlib.util.find_spec("markitdown") is not None:  # pragma: no cover - absent in CI
        pytest.skip("markitdown is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        MarkItDownParser()
    assert excinfo.value.extra == "markitdown"
