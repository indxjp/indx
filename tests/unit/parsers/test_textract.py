"""Unit tests for :class:`~indx.parsers.textract.TextractParser`.

Amazon Textract is a hosted OCR service and the ``aws`` extra (``boto3``) is absent in CI, so the
vendor SDK is faked: a stub ``boto3`` module is injected into ``sys.modules`` and the dependency
gate is neutralized. The tests then prove the adapter satisfies the
:class:`~indx.parsers.base.Parser` protocol and round-trips the fake OCR response into a valid
:class:`~indx.core.parsed.ParsedDoc` without leaking the vendor response dict. One test runs
against the real (dep-absent) environment and asserts construction raises
:class:`~indx.errors.MissingExtraError`. All tests are deterministic and offline.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from indx.errors import MissingExtraError, StageError
from indx.parsers.base import Parser
from indx.parsers.textract import TextractParser

if TYPE_CHECKING:
    from collections.abc import Iterator

# A canned Textract response: two LINE blocks plus noise the adapter must filter out.
_FAKE_BLOCKS = [
    {"BlockType": "PAGE", "Text": "ignored page block"},
    {"BlockType": "LINE", "Text": "Alpha beta gamma."},
    {"BlockType": "WORD", "Text": "Alpha"},
    {"BlockType": "LINE", "Text": "Delta epsilon zeta."},
]


class _FakeTextractClient:
    """Mimics the boto3 ``textract`` client: exposes ``detect_document_text``. A VENDOR type."""

    def __init__(self, calls: dict[str, Any]) -> None:
        self._calls = calls

    def detect_document_text(self, *, Document: dict[str, Any]) -> dict[str, Any]:
        self._calls["document"] = Document
        return {"Blocks": _FAKE_BLOCKS}


def _make_fake_boto3(calls: dict[str, Any]) -> types.ModuleType:
    """Build a fake ``boto3`` module recording how the client is constructed."""
    fake_module = types.ModuleType("boto3")

    def client(service: str, *, region_name: str | None = None) -> _FakeTextractClient:
        calls["service"] = service
        calls["region_name"] = region_name
        return _FakeTextractClient(calls)

    fake_module.client = client  # type: ignore[attr-defined]
    return fake_module


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Inject a fake ``boto3`` module and neutralize the dependency gate.

    The adapter binds ``require_extra`` at import time (``from ... import require_extra``), so the
    patch must target that bound name in the adapter module, not ``indx.utils.lazy``.
    """
    monkeypatch.setattr("indx.parsers.textract.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    calls: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "boto3", _make_fake_boto3(calls))
    yield calls


def test_satisfies_parser_protocol(fake_boto3: dict[str, Any]) -> None:
    assert isinstance(TextractParser(), Parser)


def test_parse_round_trips_into_parsed_doc(fake_boto3: dict[str, Any], tmp_path: Path) -> None:
    src = tmp_path / "scan.png"
    src.write_bytes(b"\x89PNG ignored: the fake client supplies the OCR result")

    parsed = TextractParser().parse(src)

    assert parsed.parser == "textract"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)

    kinds = [b.kind for b in parsed.blocks]
    texts = [b.text for b in parsed.blocks]
    orders = [b.order for b in parsed.blocks]
    assert kinds == ["text", "text"]
    assert texts == ["Alpha beta gamma.", "Delta epsilon zeta."]
    assert orders == [0, 1]

    # The raw file bytes are handed to Textract inline; no vendor dict escapes.
    assert fake_boto3["service"] == "textract"
    assert fake_boto3["document"] == {"Bytes": src.read_bytes()}


def test_region_comes_from_config_then_env(
    fake_boto3: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = tmp_path / "doc.png"
    src.write_bytes(b"x")

    # Explicit config region wins.
    TextractParser(region="eu-west-1").parse(src)
    assert fake_boto3["region_name"] == "eu-west-1"

    # Otherwise fall back to the AWS_REGION environment variable.
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    TextractParser().parse(src)
    assert fake_boto3["region_name"] == "ap-northeast-1"


def test_parse_rejects_oversized_document(fake_boto3: dict[str, Any], tmp_path: Path) -> None:
    src = tmp_path / "big.pdf"
    src.write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))

    with pytest.raises(StageError) as excinfo:
        TextractParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_parse_translates_vendor_failure_to_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("indx.parsers.textract.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    class _BoomClient:
        def detect_document_text(self, *, Document: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("vendor blew up")

    fake_module = types.ModuleType("boto3")
    fake_module.client = lambda *a, **k: _BoomClient()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_module)

    src = tmp_path / "doc.png"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        TextractParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real dep-absent environment the gate must fire. Guard with
    # find_spec (what require_extra itself uses) so the test skips rather than fails if a
    # contributor has the optional extra installed locally.
    if importlib.util.find_spec("boto3") is not None:  # pragma: no cover
        pytest.skip("boto3 is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        TextractParser()
    assert excinfo.value.extra == "aws"
