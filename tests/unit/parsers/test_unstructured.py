"""Unit tests for :class:`indx.parsers.unstructured.UnstructuredParser`.

The ``unstructured`` extra is not installed in the offline suite, so the vendor SDK is faked:
a stand-in ``unstructured.partition.auto`` module is injected into ``sys.modules`` and the
dependency gate is neutralized. The tests stay deterministic and fully offline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from indx.core.parsed import ParsedDoc
from indx.errors import MissingExtraError
from indx.parsers.base import Parser
from indx.parsers.unstructured import UnstructuredParser


@dataclass
class _FakeElement:
    """Minimal stand-in for an Unstructured ``Element`` (exposes ``.text`` / ``.category``)."""

    text: str
    category: str


# The elements the fake partitioner returns, in fixed order, covering every kind branch plus
# an unknown category (-> "text") and a ``None`` text value (-> "").
_FAKE_ELEMENTS = [
    _FakeElement(text="My Title", category="Title"),
    _FakeElement(text="Some narrative text.", category="NarrativeText"),
    _FakeElement(text="cell1 cell2", category="Table"),
    _FakeElement(text="Figure 1: a chart", category="FigureCaption"),
    _FakeElement(text="print('hi')", category="CodeSnippet"),
    _FakeElement(text="first bullet", category="ListItem"),
    _FakeElement(text="weird thing", category="UncategorizedText"),
]


def _install_fake_unstructured(
    monkeypatch: pytest.MonkeyPatch,
    *,
    elements: list[_FakeElement] | None = None,
    raises: Exception | None = None,
) -> dict[str, object]:
    """Inject a fake ``unstructured.partition.auto`` module and neutralize the extra gate.

    Returns a dict recording the kwargs the fake ``partition`` was called with, so a test can
    assert the adapter passed ``filename=<path>``.
    """
    captured: dict[str, object] = {}

    def _partition(*args: object, **kwargs: object) -> list[_FakeElement]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        if raises is not None:
            raise raises
        return list(_FAKE_ELEMENTS if elements is None else elements)

    auto_mod = ModuleType("unstructured.partition.auto")
    auto_mod.partition = _partition  # type: ignore[attr-defined]
    partition_pkg = ModuleType("unstructured.partition")
    root_pkg = ModuleType("unstructured")

    monkeypatch.setitem(sys.modules, "unstructured", root_pkg)
    monkeypatch.setitem(sys.modules, "unstructured.partition", partition_pkg)
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", auto_mod)
    # Neutralize the construction-time dependency gate so __init__ proceeds. The adapter does
    # ``from indx.utils.lazy import require_extra``, so the name resolved at call time lives in
    # its own module namespace; patch there (and at the source for good measure).
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.parsers.unstructured.require_extra", lambda *a, **k: None)
    return captured


def test_satisfies_parser_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_unstructured(monkeypatch)
    parser = UnstructuredParser()
    assert isinstance(parser, Parser)
    assert parser.name == "unstructured"


def test_parse_roundtrips_to_parsed_doc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _install_fake_unstructured(monkeypatch)
    src = tmp_path / "doc.html"
    src.write_text("ignored - the fake partitioner does not read it")

    parser = UnstructuredParser()
    doc = parser.parse(src)

    assert isinstance(doc, ParsedDoc)
    assert doc.source_path == str(src)
    assert doc.parser == "unstructured"
    assert doc.parser_version == "1"

    # The adapter must call partition with the path as filename=.
    assert captured["kwargs"] == {"filename": str(src)}

    # One block per element, in original order, with mapped kinds and preserved text.
    assert [b.order for b in doc.blocks] == list(range(len(_FAKE_ELEMENTS)))
    assert [(b.kind, b.text) for b in doc.blocks] == [
        ("heading", "My Title"),
        ("text", "Some narrative text."),
        ("table", "cell1 cell2"),
        ("caption", "Figure 1: a chart"),
        ("code", "print('hi')"),
        ("list", "first bullet"),
        ("text", "weird thing"),
    ]


def test_parse_handles_none_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_unstructured(
        monkeypatch,
        elements=[_FakeElement(text="", category="NarrativeText")],
    )
    # Simulate an element whose .text is None (vendor allows it).
    sys.modules["unstructured.partition.auto"].partition = (  # type: ignore[attr-defined]
        lambda *a, **k: [type("E", (), {"text": None, "category": "NarrativeText"})()]
    )
    parser = UnstructuredParser()
    doc = parser.parse(tmp_path / "f.txt")
    assert doc.blocks[0].text == ""
    assert doc.blocks[0].kind == "text"


def test_parse_translates_vendor_error_to_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from indx.errors import StageError

    _install_fake_unstructured(monkeypatch, raises=RuntimeError("boom"))
    parser = UnstructuredParser()
    with pytest.raises(StageError) as excinfo:
        parser.parse(tmp_path / "bad.xyz")
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(tmp_path / "bad.xyz")
    assert "unstructured failed" in str(excinfo.value)


def test_construction_without_extra_raises_missing_extra() -> None:
    # Do NOT neutralize require_extra: in the real dep-absent env, constructing must fail loud.
    if "unstructured" in sys.modules:  # pragma: no cover - extra not installed in CI
        pytest.skip("unstructured is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        UnstructuredParser()
    assert excinfo.value.extra == "unstructured"
