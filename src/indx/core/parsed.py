"""ParsedDoc — normalized parser output.

This is the boundary every Parser slot writes to. Because it is serializable, it can be
*recorded* once and replayed in the fast test suite without re-running a heavy parser
(testing-strategy §3.4).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Block(BaseModel):
    """One structural unit of parsed content (a paragraph, heading, table cell, caption)."""

    kind: str = "text"  # text | heading | table | caption | code | ...
    text: str = ""
    order: int = 0


class ParsedDoc(BaseModel):
    """Normalized output of parsing a single source file."""

    source_path: str
    parser: str = "plaintext"
    parser_version: str = "0"
    blocks: list[Block] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated block text in document order."""
        return "\n\n".join(b.text for b in sorted(self.blocks, key=lambda b: b.order))
