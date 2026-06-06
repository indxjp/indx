"""Source — lightweight provenance attached to chunks and parsed units.

A ``Source`` records where a piece of content came from in the walked tree (FR-DM-3,
data-models.md §Source). It travels with chunks and :class:`~indx.core.parsed.ParsedDoc`
so a :class:`~indx.store.base.SearchHit` can expose ``hit.source.path`` without
re-resolving the parent document.
"""

from __future__ import annotations

from pydantic import BaseModel


class Source(BaseModel):
    """Provenance of a chunk or parsed unit (data-models.md §Source).

    ``path`` and ``folder`` are relative to the walked root so the record stays portable
    across machines (NFR-PORT-1). ``type`` is the detected/enriched document type.
    """

    path: str
    folder: str = ""
    type: str | None = None
