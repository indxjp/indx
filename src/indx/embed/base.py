"""Embedder protocol: text -> vectors, with a declared dimension."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Batched for efficiency (technology-selections §11)."""
        ...
