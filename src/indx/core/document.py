"""Document — one enriched source file plus its directory provenance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from indx.core.relation import Relation


class Document(BaseModel):
    """A source file lifted into the knowledge space.

    ``lineage`` is the ordered list of folder names from the indexed root down to (but not
    including) the file — the structure file-level parsers throw away (PRD §2).

    The deterministic, tested storage fields are ``doc_type``/``lineage``/``size_bytes``.
    The documented read surface (data-models.md §Document) is layered on additively:
    ``type`` aliases ``doc_type``; ``folder`` joins the lineage; ``chunk_ids``/
    ``references``/``referenced_by``/``metadata`` are optional carriers with empty defaults.
    """

    id: str
    path: str  # relative to the indexed root; portable across machines (NFR-PORT-1)
    lineage: list[str] = Field(default_factory=list)
    size_bytes: int = 0

    # Enrichment (FR-S05). Populated by the Enrich stage; empty in the zero-LLM default.
    doc_type: str | None = None
    topics: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None

    # Documented additive surface (data-models.md / index-json.md). Safe empty defaults.
    chunk_ids: list[str] = Field(default_factory=list)
    references: list[Relation] = Field(default_factory=list)
    referenced_by: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def type(self) -> str | None:
        """Detected/enriched document type (data-models.md alias of ``doc_type``)."""
        return self.doc_type

    @property
    def folder(self) -> str:
        """Containing folder relative to root, joined from ``lineage`` (data-models.md)."""
        return "/".join(self.lineage)
