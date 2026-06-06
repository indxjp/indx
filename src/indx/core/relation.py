"""Relation — a typed graph edge between documents (or chunks)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class RelationType(StrEnum):
    SIBLING = "sibling"
    PARENT = "parent"
    REFERENCES = "references"
    CONTINUES = "continues"
    DUPLICATE_OF = "duplicate-of"


class Relation(BaseModel):
    """A directed, typed edge: ``src`` --type--> ``dst`` (FR-DM-5)."""

    src: str
    dst: str
    type: RelationType
    score: float = 1.0  # confidence; threshold-based types (references/duplicate-of) use it
