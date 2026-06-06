"""QwenVLClient — local Qwen2-VL vision-language model via transformers + torch.

Describes an image (optionally guided by a prompt) by running a Qwen2-VL model locally.
Requires the ``qwen-vl`` extra (``pip install indx[qwen-vl]``); the heavy ``transformers``
and ``torch`` SDKs are imported lazily inside the methods that need them so a clean
``pip install indx`` never pays for them (coding-standards §2).

Generation is deterministic (``do_sample=False``) so the same image and prompt yield a
stable caption (coding-standards §1.4).
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from PIL.Image import (  # type: ignore[import-not-found]  # optional extra: qwen-vl
        Image as PILImage,
    )

logger = logging.getLogger(__name__)

_SLOT = "vlm"
_NAME = "qwen-vl"
_EXTRA = "qwen-vl"

_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
_DEFAULT_PROMPT = "Describe this image."
_MAX_NEW_TOKENS = 512


class QwenVLClient:
    """Local Qwen2-VL adapter (satisfies the VLM protocol structurally).

    Installed via ``indx[qwen-vl]``. Loads a Qwen2-VL model and processor from
    ``transformers`` and runs inference on the local machine (CPU or GPU, auto-selected
    by ``device_map="auto"``). No network egress and no API key (coding-standards §13).

    Attributes:
        name: The registry key used in config and on the CLI.
        model: The Hugging Face model identifier to load.
    """

    name = _NAME

    def __init__(self, model: str | None = None) -> None:
        """Initialise the adapter and verify the optional extra is installed.

        ``require_extra`` is cheap (``importlib.util.find_spec``) and is the first
        statement so selecting this backend without the ``qwen-vl`` extra fails fast
        with the ``pip install indx[qwen-vl]`` hint instead of an ``ImportError`` from
        deep inside ``transformers`` (coding-standards §6.3).

        Args:
            model: Hugging Face model id to load. Defaults to a small Qwen2-VL instruct
                checkpoint when ``None``.

        Raises:
            MissingExtraError: If ``transformers`` or ``torch`` is not importable.
        """
        require_extra(_SLOT, _NAME, _EXTRA, "transformers", "torch")
        self.model = model or _DEFAULT_MODEL
        self._model: Any | None = None  # Any: vendor type, never leaks into core/
        self._processor: Any | None = None  # Any: vendor type, never leaks into core/

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Lazily load the Qwen2-VL model and processor, caching them on the instance.

        Returns:
            A ``(model, processor)`` pair of vendor objects, confined to this adapter.
        """
        if self._model is None or self._processor is None:
            # Heavy/optional imports inside the method (coding-standards §2).
            from transformers import (  # type: ignore[import-not-found]  # optional extra: qwen-vl
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )

            logger.info("vlm backend=%s loading model=%s", _NAME, self.model)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model,
                device_map="auto",
            )
            self._processor = AutoProcessor.from_pretrained(self.model)
        return self._model, self._processor

    def _decode_image(self, image: bytes) -> PILImage:
        """Decode raw image bytes into a PIL image at the adapter edge.

        Args:
            image: Raw encoded image bytes (e.g. PNG/JPEG).

        Returns:
            A decoded RGB PIL image. The PIL type stays inside this adapter and never
            reaches ``core/`` (coding-standards §6.2).
        """
        from PIL import Image  # type: ignore[import-not-found]  # optional extra: qwen-vl

        return Image.open(io.BytesIO(image)).convert("RGB")

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        """Describe an image, optionally guided by a prompt.

        Args:
            image: Raw encoded image bytes to describe.
            prompt: Optional natural-language instruction guiding the description.
                Falls back to a generic "describe this image" prompt when ``None``.

        Returns:
            The model's caption as a plain string. The vendor tensors and processor
            outputs are converted to ``str`` here and never escape the adapter.
        """
        import torch  # type: ignore[import-not-found]  # optional extra: qwen-vl

        model, processor = self._ensure_loaded()
        pil_image = self._decode_image(image)
        text_prompt = prompt or _DEFAULT_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[chat_text],
            images=[pil_image],
            return_tensors="pt",
        )
        # Co-locate CPU processor tensors with the model weights, which device_map="auto"
        # may have placed on CUDA; otherwise model.generate() raises a device-mismatch error.
        inputs = inputs.to(model.device)

        # Deterministic decoding: no sampling, no hidden randomness (coding-standards §1.4).
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

        # Strip the prompt tokens so only the freshly generated caption is decoded.
        prompt_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(prompt_ids, generated, strict=False)]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
