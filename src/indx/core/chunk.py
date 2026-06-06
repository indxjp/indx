"""Chunk — a retrievable unit that remembers where it came from and what's next to it."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from indx.core.relation import Relation
from indx.core.source import Source


class Chunk(BaseModel):
    """A slice of a Document's text, carrying provenance and neighbor links (FR-DM-4).

    The internal storage fields (``position``, ``prev_id``/``next_id``) are the tested,
    deterministic representation. The documented read surface (data-models.md §Chunk) is
    layered on top additively: ``index`` aliases ``position``; ``neighbors`` resolves to the
    adjacent ids; ``source``/``metadata``/``relations`` are optional carriers that default
    to safe empty values so existing construction stays valid.
    """

    id: str
    doc_id: str
    position: int  # 0-based order within the source document
    text: str
    # Neighbor links power context expansion and cross-file `continues` (US-1).
    prev_id: str | None = None
    next_id: str | None = None
    embedding: list[float] | None = None

    # Documented additive surface (data-models.md / index-json.md). Optional with safe
    # defaults so the deterministic pipeline can keep constructing chunks positionally.
    source: Source | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)

    @property
    def index(self) -> int:
        """0-based position within the parent document (data-models.md alias of ``position``)."""
        return self.position

    @property
    def neighbors(self) -> list[str]:
        """Adjacent chunk ids — the previous and next chunk, in order (data-models.md §Chunk).

        Derived from the stored ``prev_id``/``next_id`` so it always reflects the actual
        neighbor links the Chunk stage assigned.
        """
        return [n for n in (self.prev_id, self.next_id) if n is not None]
