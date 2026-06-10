"""Unit tests for :class:`~indx.parsers.docintel.DocumentIntelligenceParser`.

Azure AI Document Intelligence is a hosted service and the ``azure`` extra is absent in CI, so
the vendor SDK is faked: stub ``azure.ai.documentintelligence`` and ``azure.core.credentials``
modules are injected into ``sys.modules`` (every namespace parent registered) and the
dependency gate is neutralized. The tests then prove the adapter satisfies the
:class:`~indx.parsers.base.Parser` protocol and round-trips the fake cloud analysis into a
valid :class:`~indx.core.parsed.ParsedDoc` without leaking the vendor result type. One test
runs against the real (dep-absent) environment and asserts construction raises
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
from indx.parsers.docintel import DocumentIntelligenceParser

if TYPE_CHECKING:
    from collections.abc import Iterator


def _azure_documentintelligence_installed() -> bool:
    """Return whether the optional SDK is importable, never raising on a missing parent.

    ``find_spec`` on a dotted name raises ``ModuleNotFoundError`` when a namespace-package
    parent (here ``azure``) is absent, so guard it so the missing-extra test can decide to skip
    only when the SDK really is present.
    """
    try:
        return importlib.util.find_spec("azure.ai.documentintelligence") is not None
    except ModuleNotFoundError:
        return False


# Extracted content whose blank-line split yields a heading and two text paragraphs.
_FAKE_CONTENT = "# Title\n\nAlpha beta gamma.\n\nDelta epsilon zeta.\n"


class _FakeResult:
    """Mimics a Document Intelligence ``AnalyzeResult``: exposes ``.content``. A VENDOR type."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakePoller:
    """Mimics the long-running-operation poller returned by ``begin_analyze_document``."""

    def __init__(self, content: str) -> None:
        self._content = content

    def result(self) -> _FakeResult:
        return _FakeResult(self._content)


class _FakeAzureKeyCredential:
    """Mimics ``azure.core.credentials.AzureKeyCredential``. A VENDOR type."""

    def __init__(self, key: str) -> None:
        self.key = key


class _FakeDocumentIntelligenceClient:
    """Mimics ``azure.ai.documentintelligence.DocumentIntelligenceClient`` for offline tests.

    The real SDK client is a context manager that owns a long-lived HTTP transport; mirroring
    ``__enter__``/``__exit__``/``close`` here lets tests assert the adapter releases it.
    """

    def __init__(self, *, endpoint: str, credential: object) -> None:
        self.endpoint = endpoint
        self.credential = credential
        self.closed = False

    def begin_analyze_document(self, model_id: str, *, body: object) -> _FakePoller:
        # The adapter must pass the model id positionally and the file handle via ``body=``.
        assert model_id == "prebuilt-read"
        assert hasattr(body, "read")
        return _FakePoller(_FAKE_CONTENT)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _FakeDocumentIntelligenceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _install_fake_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the faked ``azure.*`` namespace packages in ``sys.modules``.

    Namespace packages need every parent present, so ``azure``, ``azure.ai``,
    ``azure.ai.documentintelligence``, ``azure.core``, and ``azure.core.credentials`` are all
    injected.
    """
    azure_mod = types.ModuleType("azure")
    azure_ai_mod = types.ModuleType("azure.ai")
    docintel_mod = types.ModuleType("azure.ai.documentintelligence")
    docintel_mod.DocumentIntelligenceClient = _FakeDocumentIntelligenceClient  # type: ignore[attr-defined]
    azure_core_mod = types.ModuleType("azure.core")
    azure_creds_mod = types.ModuleType("azure.core.credentials")
    azure_creds_mod.AzureKeyCredential = _FakeAzureKeyCredential  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.ai", azure_ai_mod)
    monkeypatch.setitem(sys.modules, "azure.ai.documentintelligence", docintel_mod)
    monkeypatch.setitem(sys.modules, "azure.core", azure_core_mod)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_creds_mod)


@pytest.fixture
def fake_docintel(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject fake ``azure.*`` modules and neutralize the dependency gate.

    The adapter binds ``require_extra`` at import time (``from ... import require_extra``), so
    the patch must target that bound name in the adapter module, not ``indx.utils.lazy``.
    """
    monkeypatch.setattr("indx.parsers.docintel.require_extra", lambda *a, **k: None)
    monkeypatch.setenv(
        "AZURE_DOCUMENTINTELLIGENCE_ENDPOINT", "https://example.cognitiveservices.azure.com"
    )
    monkeypatch.setenv("AZURE_DOCUMENTINTELLIGENCE_KEY", "di-secret")
    _install_fake_azure(monkeypatch)
    yield


def test_satisfies_parser_protocol(fake_docintel: None) -> None:
    assert isinstance(DocumentIntelligenceParser(), Parser)


def test_parse_round_trips_into_parsed_doc(fake_docintel: None, tmp_path: Path) -> None:
    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 ignored: the fake client supplies the content")

    parsed = DocumentIntelligenceParser().parse(src)

    assert parsed.parser == "docintel"
    assert parsed.parser_version == "1"
    assert parsed.source_path == str(src)

    kinds = [b.kind for b in parsed.blocks]
    texts = [b.text for b in parsed.blocks]
    orders = [b.order for b in parsed.blocks]
    assert kinds == ["heading", "text", "text"]
    assert texts == ["# Title", "Alpha beta gamma.", "Delta epsilon zeta."]
    assert orders == [0, 1, 2]


def test_parse_closes_client_releasing_transport(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: the client owns a long-lived HTTP transport and must be closed per parse,
    # otherwise a corpus of N files leaks N pooled connections / sockets.
    created: list[_FakeDocumentIntelligenceClient] = []

    class _TrackingClient(_FakeDocumentIntelligenceClient):
        def __init__(self, *, endpoint: str, credential: object) -> None:
            super().__init__(endpoint=endpoint, credential=credential)
            created.append(self)

    docintel_mod = sys.modules["azure.ai.documentintelligence"]
    docintel_mod.DocumentIntelligenceClient = _TrackingClient  # type: ignore[attr-defined]

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    DocumentIntelligenceParser().parse(src)

    assert len(created) == 1
    assert created[0].closed is True


def test_parse_closes_client_on_vendor_failure(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Even when analyze raises, the context manager must still release the transport.
    created: list[_FakeDocumentIntelligenceClient] = []

    class _BoomTrackingClient(_FakeDocumentIntelligenceClient):
        def __init__(self, *, endpoint: str, credential: object) -> None:
            super().__init__(endpoint=endpoint, credential=credential)
            created.append(self)

        def begin_analyze_document(self, model_id: str, *, body: object) -> _FakePoller:
            raise RuntimeError("vendor blew up")

    docintel_mod = sys.modules["azure.ai.documentintelligence"]
    docintel_mod.DocumentIntelligenceClient = _BoomTrackingClient  # type: ignore[attr-defined]

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError):
        DocumentIntelligenceParser().parse(src)

    assert len(created) == 1
    assert created[0].closed is True


def test_reads_endpoint_and_key_from_env(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeDocumentIntelligenceClient):
        def __init__(self, *, endpoint: str, credential: object) -> None:
            super().__init__(endpoint=endpoint, credential=credential)
            captured["endpoint"] = endpoint
            captured["credential_key"] = getattr(credential, "key", None)

    monkeypatch.setenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT", "https://env.example.com")
    monkeypatch.setenv("AZURE_DOCUMENTINTELLIGENCE_KEY", "env-key")
    docintel_mod = sys.modules["azure.ai.documentintelligence"]
    docintel_mod.DocumentIntelligenceClient = _CapturingClient  # type: ignore[attr-defined]

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    DocumentIntelligenceParser().parse(src)

    assert captured["endpoint"] == "https://env.example.com"
    assert captured["credential_key"] == "env-key"


def test_missing_endpoint_raises_stage_error(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT", raising=False)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentIntelligenceParser().parse(src)
    assert excinfo.value.stage == "parse"


def test_missing_key_raises_stage_error(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AZURE_DOCUMENTINTELLIGENCE_KEY", raising=False)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentIntelligenceParser().parse(src)
    assert excinfo.value.stage == "parse"


def test_parse_translates_vendor_failure_to_stage_error(
    fake_docintel: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BoomClient(_FakeDocumentIntelligenceClient):
        def begin_analyze_document(self, model_id: str, *, body: object) -> _FakePoller:
            raise RuntimeError("vendor blew up")

    docintel_mod = sys.modules["azure.ai.documentintelligence"]
    docintel_mod.DocumentIntelligenceClient = _BoomClient  # type: ignore[attr-defined]

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    with pytest.raises(StageError) as excinfo:
        DocumentIntelligenceParser().parse(src)
    assert excinfo.value.stage == "parse"
    assert excinfo.value.path == str(src)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization: in the real (dep-absent) environment the gate must fire. The
    # require_extra helper maps a missing namespace parent (no ``azure`` package) to a clean
    # MissingExtraError, so this works even though ``azure.ai.documentintelligence`` is dotted.
    if _azure_documentintelligence_installed():  # pragma: no cover
        pytest.skip("azure-ai-documentintelligence installed; cannot exercise missing-extra path")
    with pytest.raises(MissingExtraError) as excinfo:
        DocumentIntelligenceParser()
    assert excinfo.value.extra == "azure"
