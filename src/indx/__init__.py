"""indx — make directories AI-ready, not just files.

Public SDK surface. The CLI is this same surface with handles (CLI⇄SDK parity).
"""

from indx._version import __version__
from indx.core import (
    Chunk,
    Document,
    KnowledgeSpace,
    Manifest,
    ParsedDoc,
    Relation,
    RelationType,
    Source,
    SpaceContext,
    SpaceStats,
)
from indx.pipeline import DirectoryPipeline
from indx.store.base import SearchHit

__all__ = [
    "__version__",
    "DirectoryPipeline",
    "KnowledgeSpace",
    "Manifest",
    "Document",
    "Chunk",
    "Relation",
    "RelationType",
    "ParsedDoc",
    "Source",
    "SearchHit",
    "SpaceContext",
    "SpaceStats",
]
