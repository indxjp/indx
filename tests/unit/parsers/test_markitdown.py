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


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real dep-absent environment the gate must fire. Guard with
    # find_spec (what require_extra itself uses) so the test skips rather than fails if a
    # contributor has the optional extra installed locally.
    if importlib.util.find_spec("markitdown") is not None:  # pragma: no cover - absent in CI
        pytest.skip("markitdown is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        MarkItDownParser()
    assert excinfo.value.extra == "markitdown"
