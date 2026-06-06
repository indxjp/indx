"""Parser protocol: turn a source file into a normalized ParsedDoc."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from indx.core.parsed import ParsedDoc


@runtime_checkable
class Parser(Protocol):
    """Contract for every parser slot implementation."""

    name: str

    def parse(self, path: Path) -> ParsedDoc:
        """Parse one file into a :class:`ParsedDoc`. Local parsers must not touch the network."""
        ...
