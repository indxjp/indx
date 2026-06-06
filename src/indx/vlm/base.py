"""VLM protocol: vision-language enrichment for images/figures. Default: ``none`` (disabled).

See technical-spec §4.3. Implementations describe an image (optionally guided by a prompt).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VLM(Protocol):
    name: str

    def describe(self, image: bytes, *, prompt: str | None = None) -> str: ...
