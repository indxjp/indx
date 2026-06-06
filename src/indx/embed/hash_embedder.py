"""HashEmbedder — deterministic, zero-dependency embedder.

Hashes token features into a fixed-dim L2-normalized vector. Not semantically strong, but
**fully deterministic across machines** — which makes it the default for the offline test
suite (testing-strategy §3.7) and a real air-gapped fallback that needs no model download.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"\w+")


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # noqa: S324 (not security)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
