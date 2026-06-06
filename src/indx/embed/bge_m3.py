"""BGEM3Embedder — default real embedder (``BAAI/bge-m3``). Requires the `bge` extra.

Loads BGE-M3 locally via FlagEmbedding's ``BGEM3FlagModel`` (which pulls in torch), so the
heavy import lives behind the ``indx[bge]`` extra and is **never** imported at module top
level (coding-standards §2, §6.3). Only the model's *dense* output is used: a fixed
1024-dim, L2-normalized vector per input, in input order (reference/07-embeddings.md). This
is a local-profile embedder — no API key, no network at inference time once weights are
cached.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from indx.errors import StageError
from indx.utils.lazy import require_extra

if TYPE_CHECKING:
    from FlagEmbedding import (  # type: ignore[import-not-found]  # optional extra: bge
        BGEM3FlagModel,
    )

_logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """Local BGE-M3 dense embedder backed by FlagEmbedding.

    Turns a batch of chunk texts into 1024-dim dense vectors. The full batch is handed to a
    single ``encode`` call so the model vectorizes it in one pass (batching is the key perf
    lever; reference/07-embeddings.md). Output is deterministic by default: ``use_fp16`` is
    off because fp16 changes the emitted bytes and is GPU-nondeterministic.

    Attributes:
        name: Stable model identity recorded into the archive manifest.
        dim: Fixed dense dimensionality of BGE-M3 (1024).
    """

    name = "bge-m3"
    dim = 1024  # BGE-M3 dense dimension (fixed)

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        *,
        batch_size: int = 12,
        max_length: int = 8192,
        use_fp16: bool = False,
        device: str | None = None,
    ) -> None:
        """Load the BGE-M3 model, failing fast if the ``bge`` extra is absent.

        Args:
            model_id: Hugging Face model id to load. Recorded as ``name`` in the manifest.
            batch_size: Sub-batch size passed to ``encode`` for the local forward pass.
            max_length: Maximum token length per input (BGE-M3 supports up to 8192).
            use_fp16: Use half precision. Off by default: fp16 changes output bytes and is
                GPU-nondeterministic, which breaks byte-stable golden files (constraint 4).
            device: Optional device string (e.g. ``"cuda:0"``); ``None`` lets FlagEmbedding
                pick.

        Raises:
            MissingExtraError: If the ``bge`` extra (FlagEmbedding) is not installed.
        """
        # Fail with one actionable message if the extra is absent. Must be the first
        # statement so importing this module stays safe (coding-standards §6.3).
        require_extra("embedder", "bge-m3", "bge", "FlagEmbedding")
        from FlagEmbedding import BGEM3FlagModel  # optional extra: bge

        self.name = model_id  # record the *actual* model in the manifest
        self.batch_size = batch_size
        self.max_length = max_length
        devices = [device] if device else None
        self._model: BGEM3FlagModel = BGEM3FlagModel(model_id, use_fp16=use_fp16, devices=devices)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense 1024-dim vectors, in input order.

        Args:
            texts: Batch of input strings. May be empty.

        Returns:
            One ``list[float]`` of length :attr:`dim` per input text, in input order. An
            empty input yields an empty list.

        Raises:
            StageError: If the backend ``encode`` call fails.
        """
        if not texts:
            return []
        try:
            out: dict[str, Any] = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # normalize backend errors to a typed IndxError
            raise StageError("embed", str(exc)) from exc
        # FlagEmbedding returns a dict; dense_vecs is an ndarray (n, 1024).
        return [list(map(float, row)) for row in out["dense_vecs"]]
