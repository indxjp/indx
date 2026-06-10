"""Regression test: ``DocumentAIParser`` honors the ``GOOGLE_CLOUD_LOCATION`` env fallback.

The constructor's documented contract is that ``location`` falls back to
``GOOGLE_CLOUD_LOCATION`` (and finally to ``"us"``). A previous truthy default
(``location: str = "us"``) short-circuited the ``or`` chain so the env-var branch was
unreachable dead code. With ``location: str | None = None`` the fallback fires: an ``eu``
processor must resolve to ``locations/eu`` and the ``eu-documentai.googleapis.com`` endpoint.

Fully offline: the vendor ``google.cloud.documentai`` SDK is faked and the extra gate is
neutralized, mirroring ``tests/unit/parsers/test_docai.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from indx.parsers.docai import DocumentAIParser

if TYPE_CHECKING:
    from collections.abc import Iterator

_FAKE_TEXT = "Alpha beta gamma.\n"


class _FakeRawDocument:
    def __init__(self, *, content: bytes, mime_type: str) -> None:
        self.content = content
        self.mime_type = mime_type


class _FakeProcessRequest:
    def __init__(self, *, name: str, raw_document: _FakeRawDocument) -> None:
        self.name = name
        self.raw_document = raw_document


class _FakeDocument:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.document = _FakeDocument(text)


class _FakeEuClient:
    """Asserts the endpoint targets the ``eu`` multi-region (env-var fallback)."""

    last_request: _FakeProcessRequest | None = None
    last_endpoint: str | None = None

    def __init__(self, *, client_options: dict[str, str] | None = None) -> None:
        type(self).last_endpoint = (client_options or {}).get("api_endpoint")

    def processor_path(self, project: str, location: str, processor_id: str) -> str:
        return f"projects/{project}/locations/{location}/processors/{processor_id}"

    def process_document(self, *, request: _FakeProcessRequest) -> _FakeResult:
        type(self).last_request = request
        return _FakeResult(_FAKE_TEXT)


def _make_documentai_module() -> types.ModuleType:
    module = types.ModuleType("google.cloud.documentai")
    module.DocumentProcessorServiceClient = _FakeEuClient  # type: ignore[attr-defined]
    module.ProcessRequest = _FakeProcessRequest  # type: ignore[attr-defined]
    module.RawDocument = _FakeRawDocument  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_documentai(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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
    _FakeEuClient.last_request = None
    _FakeEuClient.last_endpoint = None
    yield


def test_location_falls_back_to_env_var(
    fake_documentai: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "eu")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    DocumentAIParser(processor_id="proc").parse(src)

    request = _FakeEuClient.last_request
    assert request is not None
    assert request.name == "projects/env-proj/locations/eu/processors/proc"
    assert _FakeEuClient.last_endpoint == "eu-documentai.googleapis.com"


def test_explicit_location_overrides_env(
    fake_documentai: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    DocumentAIParser(location="eu", processor_id="proc").parse(src)

    request = _FakeEuClient.last_request
    assert request is not None
    assert request.name == "projects/env-proj/locations/eu/processors/proc"
    assert _FakeEuClient.last_endpoint == "eu-documentai.googleapis.com"
