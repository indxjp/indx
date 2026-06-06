"""NullVLM — the default ``none`` VLM: vision enrichment disabled. Ships in core, zero-dep."""

from __future__ import annotations


class NullVLM:
    name = "none"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def describe(self, image: bytes, *, prompt: str | None = None) -> str:
        return ""
