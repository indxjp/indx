"""indx domain model — Pydantic v2 + stdlib only. Depends on nothing else internal."""

from indx.core.chunk import Chunk
from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.knowledge_space import Answer, ChildRef, KnowledgeSpace, Manifest
from indx.core.parsed import Block, ParsedDoc
from indx.core.relation import Relation, RelationType
from indx.core.source import Source
from indx.core.stats import SpaceStats

__all__ = [
    "Answer",
    "Block",
    "ChildRef",
    "Chunk",
    "Document",
    "KnowledgeSpace",
    "Manifest",
    "ParsedDoc",
    "Relation",
    "RelationType",
    "Source",
    "SpaceContext",
    "SpaceStats",
]
