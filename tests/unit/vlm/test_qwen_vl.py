"""Mocked unit tests for QwenVLClient — transformers/torch/PIL are absent, so they are faked.

Fake ``transformers``, ``torch`` and ``PIL`` modules are injected into ``sys.modules`` and the
dependency gate is neutralized so construction proceeds offline (coding-standards §11). The real
dep-absent construction is asserted in a separate test that does *not* neutralize the gate.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from indx.errors import MissingExtraError
from indx.vlm.base import VLM


class _FakeProcessorOutput(dict[str, Any]):
    """Mimics a transformers BatchFeature: dict-like with the keys the adapter reads."""

    def to(self, device: Any) -> _FakeProcessorOutput:
        """Real BatchFeature.to(device) moves tensors and returns self; no-op here."""
        return self


class _FakeProcessor:
    """Stand-in for a Qwen2-VL processor.

    Records the calls the adapter makes and returns deterministic, structurally-valid
    objects so ``describe`` can round-trip to a plain string.
    """

    last_messages: list[dict[str, Any]] | None = None
    last_text: list[str] | None = None
    last_images: list[Any] | None = None

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> _FakeProcessor:
        inst = cls()
        inst.model = model
        return inst

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        type(self).last_messages = messages
        return "CHAT_TEMPLATE"

    def __call__(
        self, *, text: list[str], images: list[Any], return_tensors: str
    ) -> _FakeProcessorOutput:
        type(self).last_text = text
        type(self).last_images = images
        # input_ids is a list (one row per prompt); the adapter slices each generated row by len.
        return _FakeProcessorOutput(input_ids=[[1, 2, 3]])

    def batch_decode(
        self, sequences: list[Any], *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> list[str]:
        return ["  a red square on a white background  "]


class _FakeModel:
    """Stand-in for Qwen2VLForConditionalGeneration."""

    last_generate_kwargs: dict[str, Any] | None = None
    # Real Qwen2VLForConditionalGeneration exposes .device; the adapter reads it to
    # co-locate processor tensors with the model weights.
    device: str = "cpu"

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> _FakeModel:
        inst = cls()
        inst.model = model
        return inst

    def generate(self, **kwargs: Any) -> list[list[int]]:
        type(self).last_generate_kwargs = kwargs
        # Returns generated rows longer than the 3-token prompt so the slice yields new tokens.
        return [[1, 2, 3, 42, 43]]


class _FakeImage:
    def convert(self, mode: str) -> _FakeImage:
        return self


def _make_torch() -> types.ModuleType:
    fake = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    fake.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake transformers/torch/PIL modules and neutralize the require_extra gate."""
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = _FakeProcessor  # type: ignore[attr-defined]
    transformers.Qwen2VLForConditionalGeneration = _FakeModel  # type: ignore[attr-defined]

    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.open = lambda buf: _FakeImage()  # type: ignore[attr-defined]
    pil.Image = pil_image  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", _make_torch())
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)
    # Neutralize the dependency gate at the name the adapter actually calls. The adapter
    # binds ``require_extra`` into its own namespace (``from indx.utils.lazy import ...``),
    # so patching it there is what takes effect — and monkeypatch restores it cleanly so the
    # later dep-absent test still sees the real gate.
    monkeypatch.setattr("indx.vlm.qwen_vl.require_extra", lambda *a, **k: None)


def test_satisfies_vlm_protocol(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    assert isinstance(QwenVLClient(), VLM)
    assert QwenVLClient().name == "qwen-vl"


def test_default_and_override_model(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    assert QwenVLClient().model == "Qwen/Qwen2-VL-2B-Instruct"
    assert QwenVLClient("Qwen/Qwen2-VL-7B-Instruct").model == "Qwen/Qwen2-VL-7B-Instruct"


def test_describe_returns_caption_text(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    out = QwenVLClient().describe(b"\x89PNG fake bytes", prompt="What is this?")
    # decoded text is trimmed of surrounding whitespace at the adapter edge.
    assert out == "a red square on a white background"


def test_describe_is_deterministic_no_sampling(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    vlm = QwenVLClient()
    vlm.describe(b"img-bytes", prompt="caption please")

    assert _FakeModel.last_generate_kwargs is not None
    assert _FakeModel.last_generate_kwargs["do_sample"] is False  # determinism (§1.4)

    # The prompt text flows into the chat template messages.
    assert _FakeProcessor.last_messages is not None
    content = _FakeProcessor.last_messages[0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image")
    assert text_part["text"] == "caption please"
    assert image_part == {"type": "image"}


def test_describe_uses_default_prompt_when_none(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    QwenVLClient().describe(b"img-bytes")

    assert _FakeProcessor.last_messages is not None
    content = _FakeProcessor.last_messages[0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    assert text_part["text"] == "Describe this image."


def test_model_and_processor_are_cached(fake_sdk: None) -> None:
    from indx.vlm.qwen_vl import QwenVLClient

    vlm = QwenVLClient()
    m1, p1 = vlm._ensure_loaded()
    m2, p2 = vlm._ensure_loaded()
    assert m1 is m2
    assert p1 is p2


def test_construction_without_extra_raises_missing_extra() -> None:
    """In the real dep-absent environment, construction must raise MissingExtraError.

    This test deliberately does NOT neutralize ``require_extra``.
    """
    from indx.vlm.qwen_vl import QwenVLClient

    with pytest.raises(MissingExtraError) as exc_info:
        QwenVLClient()
    assert exc_info.value.extra == "qwen-vl"
