"""SpaceContext — the mutable, typed state threaded through every pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from indx.core.knowledge_space import KnowledgeSpace
from indx.core.parsed import ParsedDoc


class StageErrorRecord(BaseModel):
    """A non-fatal, per-item failure recorded on the context (technical-spec §3.4).

    A stage that skips one item appends a record with ``kind="skip"``; the pipeline's
    ``strict`` mode promotes any such record to a fatal :class:`~indx.errors.StageError`.
    """

    stage: str
    kind: Literal["skip", "fatal"] = "skip"
    item: str | None = None  # the offending file path or chunk id
    message: str = ""


class SpaceContext(BaseModel):
    """Shared run-state. Each stage reads what earlier stages wrote and adds its own (FR-S-CTX).

    Stages are swappable precisely because they communicate only through this object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    # When the build input was a ``.zip``, ``root`` is a temp extraction dir and ``tmp_root``
    # holds it so the pipeline can remove it once the run (or dry-run) ends. ``None`` for a
    # directory input. See ``utils/zip_input.py``.
    tmp_root: Path | None = None
    seed: int = 0  # deterministic seed for any sampling/ordering (NFR-DET-1)
    space: KnowledgeSpace = Field(default_factory=KnowledgeSpace)
    parsed: dict[str, ParsedDoc] = Field(default_factory=dict)  # doc_id -> ParsedDoc
    # Per-item, non-fatal failures (technical-spec §3.4). ``--strict`` promotes skips to fatal.
    errors: list[StageErrorRecord] = Field(default_factory=list)
