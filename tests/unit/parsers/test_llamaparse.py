"""Unit tests for :class:`~indx.parsers.llamaparse.LlamaParseParser`.

LlamaParse is a hosted service and the ``llamaparse`` extra is absent in CI, so the vendor
SDK is faked: a stub ``llama_cloud_services`` module is injected into ``sys.modules`` and the
dependency gate is neutralized. The tests then prove the adapter satisfies the
:class:`~indx.parsers.base.Parser` protocol and round-trips the fake cloud conversion into a
valid :class:`~indx.core.parsed.ParsedDoc` without leaking the vendor ``Document`` type. One
test runs against the real (dep-absent) environment and asserts construction raises
:class:`~indx.errors.MissingExtraError`. All tests are deterministic and offline.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from indx.errors import MissingExtraError, StageError
from indx.parsers.base import Parser
from indx.parsers.llamaparse import LlamaParseParser

if TYPE_CHECKING:
    from collections.abc import Iterator

# Two vendor Documents whose markdown together yields a heading and two text paragraphs.
_FAKE_DOC_TEXTS = ["# Title\n\nAlpha beta gamma.", "Delta epsilon zeta.\n"]


class _FakeDocument:
    """Mimics a LlamaParse ``Document``: exposes ``.text``. A VENDOR type."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLlamaParse:
    """Mimics ``llama_cloud_services.LlamaParse`` for offline tests."""

    def __init__(self, *, api_key: str | None = None, result_type: str = "markdown") -> None:
        # The adapter must request markdown and pass the env-sourced key (here: None).
        assert result_type == "markdown"
        self.api_key = api_key
        self.result_type = result_type

    def load_data(self, file_path: str) -> list[_FakeDocument]:
        return [_FakeDocument(t) for t in _FAKE_DOC_TEXTS]


@pytest.fixture
def fake_llama_parse(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject a fake ``llama_cloud_services`` module and neutralize the dependency gate.

    The adapter binds ``require_extra`` at import time (``from ... import require_extra``), so
    the patch must target that bound name in the adapter module, not ``indx.utils.lazy``.
    """
    monkeypatch.setattr("indx.parsers.llamaparse.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    fake_module = types.ModuleType("llama_cloud_services")
    fake_module.LlamaParse = _FakeLlamaParse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cloud_services", fake_module)
    yield


def test_satisfies_parser_protocol(fake_llama_parse: None) -> None:
    assert isinstance(LlamaParseParser(), Parser)


def test_parse_round_trips_into_parsed_doc(fake_llama_parse: None, tmp_path: Path) -> None:
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 ignored: the fake client supplies the markdown")

    parsed = LlamaParseParser().parse(src)

    assert parsed.parser == "llamaparse"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)

    kinds = [b.kind for b in parsed.blocks]
    texts = [b.text for b in parsed.blocks]
    orders = [b.order for b in parsed.blocks]
    assert kinds == ["heading", "text", "text"]
    assert texts == ["# Title", "Alpha beta gamma.", "Delta epsilon zeta."]
    assert orders == [0, 1, 2]


def test_reads_api_key_from_env(
    fake_llama_parse: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str | None] = {}

    class _CapturingLlamaParse(_FakeLlamaParse):
        def __init__(self, *, api_key: str | None = None, result_type: str = "markdown") -> None:
            super().__init__(api_key=api_key, result_type=result_type)
            captured["api_key"] = api_key

    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx-secret")
    fake_module = sys.modules["llama_cloud_services"]
    fake_module.LlamaParse = _CapturingLlamaParse  # type: ignore[attr-defined]

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    LlamaParseParser().parse(src)

    assert captured["api_key"] == "llx-secret"


def test_parse_translates_vendor_failure_to_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("indx.parsers.llamaparse.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)

    class _BoomLlamaParse:
        def __init__(self, *, api_key: str | None = None, result_type: str = "markdown") -> None:
            pass

        def load_data(self, file_path: str) -> list[_FakeDocument]:
            raise RuntimeError("vendor blew up")

    fake_module = types.ModuleType("llama_cloud_services")
    fake_module.LlamaParse = _BoomLlamaParse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cloud_services", fake_module)

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        LlamaParseParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real dep-absent environment the gate must fire. Guard with
    # find_spec (what require_extra itself uses) so the test skips rather than fails if a
    # contributor has the optional extra installed locally.
    if importlib.util.find_spec("llama_cloud_services") is not None:  # pragma: no cover
        pytest.skip("llama_cloud_services is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        LlamaParseParser()
    assert excinfo.value.extra == "llamaparse"
