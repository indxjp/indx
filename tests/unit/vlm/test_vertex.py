"""Mocked unit tests for VertexVLM — the google-genai SDK is absent, so it is faked.

The vendor ``google.genai`` namespace package is injected into ``sys.modules`` (every
parent registered: ``google``, ``google.genai``, ``google.genai.types``) and the
dependency gate is neutralized so construction proceeds offline (coding-standards §11).
The real dep-absent construction is asserted in a separate test that does *not*
neutralize the gate.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError
from indx.vlm.base import VLM


class _FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self._client.last_kwargs = kwargs
        return _FakeResponse(self._client.reply)


class _FakeClient:
    reply: str | None = "a red square on a white background"

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
    """Inject a fake ``google.genai`` namespace and neutralize the require_extra gate.

    For namespace packages every parent must be registered in ``sys.modules`` (``google``,
    ``google.genai``, ``google.genai.types``). The gate is neutralized on both the lazy
    helper and the adapter module's bound name; monkeypatch restores them on teardown so
    the dep-absent test still sees the real gate.
    """
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


def test_satisfies_vlm_protocol(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    assert isinstance(VertexVLM(), VLM)
    assert VertexVLM().name == "vertex"


def test_default_model_is_gemini_flash(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    assert VertexVLM().model == "gemini-2.5-flash"
    assert VertexVLM("gemini-2.5-pro").model == "gemini-2.5-pro"


def test_describe_returns_caption_text(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    out = vlm.describe(b"\x89PNG fake bytes", prompt="What is this?")

    assert out == "a red square on a white background"


def test_describe_builds_inline_image_part(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    image = b"\x89PNG fake bytes"
    vlm.describe(image, prompt="caption please")

    client = vlm._ensure_client()
    kwargs = client.last_kwargs
    assert kwargs["model"] == "gemini-2.5-flash"

    contents = kwargs["contents"]
    assert contents[0] == "caption please"
    part = contents[1]
    assert isinstance(part, _FakePart)
    assert part.data == image
    assert part.mime_type == "image/png"


def test_describe_uses_default_prompt(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    vlm = VertexVLM()
    vlm.describe(b"x")

    contents = vlm._ensure_client().last_kwargs["contents"]
    assert contents[0] == "Describe this image in detail."


def test_describe_handles_empty_text(
    monkeypatch: pytest.MonkeyPatch, fake_genai: type[_FakeClient]
) -> None:
    from indx.vlm.vertex import VertexVLM

    monkeypatch.setattr(_FakeClient, "reply", None)
    assert VertexVLM().describe(b"x") == ""


def test_client_built_in_vertex_mode(fake_genai: type[_FakeClient]) -> None:
    from indx.vlm.vertex import VertexVLM

    client = VertexVLM(project="proj", location="us-central1")._ensure_client()
    assert client.vertexai is True
    assert client.project == "proj"
    assert client.location == "us-central1"


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.vlm.vertex import VertexVLM

    # Guard against fake-module leakage from other tests: the real env has no google SDK.
    # ``google`` is a namespace package, so absence of the parent is what we assert; a
    # ``find_spec("google.genai")`` would raise once the parent is gone, which is itself
    # proof the SDK is unavailable.
    assert "google" not in sys.modules
    assert "google.genai" not in sys.modules
    assert importlib.util.find_spec("google") is None

    with pytest.raises(MissingExtraError) as exc_info:
        VertexVLM()
    assert exc_info.value.extra == "gcp"
