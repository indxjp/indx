"""Writer protocol: serialize a KnowledgeSpace to a destination for downstream use."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from indx.core.knowledge_space import KnowledgeSpace


@runtime_checkable
class Writer(Protocol):
    name: str

    def write(self, space: KnowledgeSpace, dest: Path, *, name: str = "handbook") -> None: ...
