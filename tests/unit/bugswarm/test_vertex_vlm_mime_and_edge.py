"""Regression tests for VertexVLM: MIME sniffing and edge error normalization.

Two confirmed bugs are covered:

1. ``describe`` hard-coded ``mime_type="image/png"`` for arbitrary image bytes, so a
   JPEG/GIF/WebP figure was mislabeled and Gemini could reject or mis-decode it. The fix
   reuses :func:`indx.vlm.gpt4o._sniff_mime`.
2. ``describe`` did not wrap the vendor call, so raw ``google.genai`` exceptions leaked
   past the edge instead of being normalized to :class:`~indx.errors.StageError`.

The vendor ``google.genai`` namespace is faked into ``sys.modules`` and the dependency
gate is neutralized so everything runs offline (coding-standards §11).
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

import pytest

from indx.errors import StageError


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self._client.last_kwargs = kwargs
        if self._client.raise_exc is not None:
            raise self._client.raise_exc
        return _FakeResponse(self._client.reply)


class _FakeClient:
    reply: str | None = "a caption"
    raise_exc: BaseException | None = None

    def __init__(
        self, *, vertexai: bool = False, project: str | None = None, location: str | None = None
    ) -> None:
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self.last_kwargs: dict[str, Any] = {}
        self.models = _FakeModels(self)


class _FakePart:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type

    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str) -> _FakePart:
        return _FakePart(data, mime_type)


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Inject a fake ``google.genai`` namespace and neutralize the require_extra gate."""
    import indx.vlm.vertex as vertex_mod

    google_mod = types.ModuleType("google")
    google_mod.__spec__ = importlib.machinery.ModuleSpec("google", loader=None)
    google_mod.__path__ = []  # type: ignore[attr-defined]  # mark as a package

    genai_mod = types.ModuleType("google.genai")
    genai_mod.__spec__ = importlib.machinery.ModuleSpec("google.genai", loader=None)
    genai_mod.Client = _FakeClient  # type: ignore[attr-defined]

    types_mod = types.ModuleType("google.genai.types")
    types_mod.__spec__ = importlib.machinery.ModuleSpec("google.genai.types", loader=None)
    types_mod.Part = _FakePart  # type: ignore[attr-defined]

    genai_mod.types = types_mod  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr(vertex_mod, "require_extra", lambda *a, **k: None)
    return _FakeClient


def test_describe_sniffs_jpeg_mime(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF rest of jpeg"
    vlm.describe(jpeg)

    part = vlm._ensure_client().last_kwargs["contents"][1]
    assert isinstance(part, _FakePart)
    assert part.mime_type == "image/jpeg"


def test_describe_sniffs_webp_mime(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    vlm.describe(webp)

    part = vlm._ensure_client().last_kwargs["contents"][1]
    assert part.mime_type == "image/webp"


def test_describe_png_still_defaults_to_png(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    vlm.describe(b"\x89PNG\r\n\x1a\n fake png")

    part = vlm._ensure_client().last_kwargs["contents"][1]
    assert part.mime_type == "image/png"


def test_describe_wraps_vendor_error_as_stage_error(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    monkeypatch_exc = RuntimeError("quota exceeded")
    fake_genai.raise_exc = monkeypatch_exc
    try:
        with pytest.raises(StageError) as exc_info:
            VertexVLM().describe(b"\x89PNG\r\n\x1a\n x")
    finally:
        fake_genai.raise_exc = None

    assert exc_info.value.stage == "enrich"
    assert exc_info.value.__cause__ is monkeypatch_exc
    assert "vertex (vlm) failed" in str(exc_info.value)
