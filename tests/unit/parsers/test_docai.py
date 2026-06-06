"""Unit tests for :class:`~indx.parsers.docai.DocumentAIParser`.

Google Document AI is a hosted service and the ``gcp`` extra is absent in CI, so the vendor
SDK is faked: a stub ``google.cloud.documentai`` namespace module is injected into
``sys.modules`` and the dependency gate is neutralized. The tests then prove the adapter
satisfies the :class:`~indx.parsers.base.Parser` protocol and round-trips the fake cloud
extraction into a valid :class:`~indx.core.parsed.ParsedDoc` without leaking the vendor
``Document`` type. One test runs against the real (dep-absent) environment and asserts
construction raises :class:`~indx.errors.MissingExtraError`. All tests are deterministic and
offline.
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
from indx.parsers.docai import DocumentAIParser

if TYPE_CHECKING:
    from collections.abc import Iterator

# Document text that yields two text paragraphs after blank-line splitting.
_FAKE_TEXT = "Alpha beta gamma.\n\nDelta epsilon zeta.\n"


class _FakeRawDocument:
    """Mimics ``documentai.RawDocument``. A VENDOR type."""

    def __init__(self, *, content: bytes, mime_type: str) -> None:
        self.content = content
        self.mime_type = mime_type


class _FakeProcessRequest:
    """Mimics ``documentai.ProcessRequest``. A VENDOR type."""

    def __init__(self, *, name: str, raw_document: _FakeRawDocument) -> None:
        self.name = name
        self.raw_document = raw_document


class _FakeDocument:
    """Mimics ``result.document``: exposes ``.text``. A VENDOR type."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    """Mimics the ``process_document`` result. A VENDOR type."""

    def __init__(self, text: str) -> None:
        self.document = _FakeDocument(text)


class _FakeClient:
    """Mimics ``documentai.DocumentProcessorServiceClient`` for offline tests."""

    last_request: _FakeProcessRequest | None = None

    def __init__(self, *, client_options: dict[str, str] | None = None) -> None:
        # The adapter must point the endpoint at the configured multi-region.
        assert client_options == {"api_endpoint": "us-documentai.googleapis.com"}
        self.client_options = client_options

    def processor_path(self, project: str, location: str, processor_id: str) -> str:
        return f"projects/{project}/locations/{location}/processors/{processor_id}"

    def process_document(self, *, request: _FakeProcessRequest) -> _FakeResult:
        type(self).last_request = request
        return _FakeResult(_FAKE_TEXT)


def _make_documentai_module(client_cls: type = _FakeClient) -> types.ModuleType:
    """Build a fake ``google.cloud.documentai`` module exposing the vendor symbols."""
    module = types.ModuleType("google.cloud.documentai")
    module.DocumentProcessorServiceClient = client_cls  # type: ignore[attr-defined]
    module.ProcessRequest = _FakeProcessRequest  # type: ignore[attr-defined]
    module.RawDocument = _FakeRawDocument  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_documentai(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject a fake ``google.cloud.documentai`` namespace and neutralize the gate.

    The adapter binds ``require_extra`` at import time (``from ... import require_extra``), so
    the patch must target that bound name in the adapter module, not ``indx.utils.lazy``. For
    the namespace package every parent must be present in ``sys.modules``.
    """
    monkeypatch.setattr("indx.parsers.docai.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    google_mod = types.ModuleType("google")
    cloud_mod = types.ModuleType("google.cloud")
    documentai_mod = _make_documentai_module()
    cloud_mod.documentai = documentai_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.documentai", documentai_mod)
    _FakeClient.last_request = None
    yield


def test_satisfies_parser_protocol(fake_documentai: None) -> None:
    assert isinstance(DocumentAIParser(project="p", processor_id="proc"), Parser)


def test_parse_round_trips_into_parsed_doc(fake_documentai: None, tmp_path: Path) -> None:
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 ignored: the fake client supplies the text")

    parsed = DocumentAIParser(project="my-proj", processor_id="proc-123").parse(src)

    assert parsed.parser == "docai"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)

    kinds = [b.kind for b in parsed.blocks]
    texts = [b.text for b in parsed.blocks]
    orders = [b.order for b in parsed.blocks]
    assert kinds == ["text", "text"]
    assert texts == ["Alpha beta gamma.", "Delta epsilon zeta."]
    assert orders == [0, 1]

    # The request carried the processor path and the PDF mime type — no vendor type escaped.
    request = _FakeClient.last_request
    assert request is not None
    assert request.name == "projects/my-proj/locations/us/processors/proc-123"
    assert request.raw_document.mime_type == "application/pdf"


def test_mime_type_inferred_from_suffix(fake_documentai: None, tmp_path: Path) -> None:
    src = tmp_path / "scan.png"
    src.write_bytes(b"\x89PNG ignored")

    DocumentAIParser(project="p", processor_id="proc").parse(src)

    request = _FakeClient.last_request
    assert request is not None
    assert request.raw_document.mime_type == "image/png"


def test_reads_project_and_location_from_env(
    fake_documentai: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    DocumentAIParser(processor_id="proc").parse(src)

    request = _FakeClient.last_request
    assert request is not None
    assert request.name == "projects/env-proj/locations/us/processors/proc"


def test_missing_project_raises_stage_error(fake_documentai: None, tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentAIParser(processor_id="proc").parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_missing_processor_id_raises_stage_error(fake_documentai: None, tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentAIParser(project="p").parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_parse_translates_vendor_failure_to_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("indx.parsers.docai.require_extra", lambda *a, **k: None)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    class _BoomClient(_FakeClient):
        def process_document(self, *, request: _FakeProcessRequest) -> _FakeResult:
            raise RuntimeError("vendor blew up")

    google_mod = types.ModuleType("google")
    cloud_mod = types.ModuleType("google.cloud")
    documentai_mod = _make_documentai_module(_BoomClient)
    cloud_mod.documentai = documentai_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.documentai", documentai_mod)

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentAIParser(project="p", processor_id="proc").parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real dep-absent environment the gate must fire. Guard with
    # find_spec (what require_extra itself uses) so the test skips rather than fails if a
    # contributor has the optional extra installed locally. For a namespace package an absent
    # parent makes find_spec raise ModuleNotFoundError, which is itself proof the dep is absent.
    try:
        installed = importlib.util.find_spec("google.cloud.documentai") is not None
    except ModuleNotFoundError:
        installed = False
    if installed:  # pragma: no cover
        pytest.skip("google.cloud.documentai is installed; cannot exercise the missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        DocumentAIParser(project="p", processor_id="proc")
    assert excinfo.value.extra == "gcp"
