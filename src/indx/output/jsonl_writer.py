"""JsonlWriter — newline-delimited documents/chunks/relations. Zero-dep, ships in core."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from indx.core.knowledge_space import KnowledgeSpace


class JsonlWriter:
    name = "jsonl"

    def write(self, space: KnowledgeSpace, dest: Path, *, name: str = "handbook") -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        _dump(dest / "documents.jsonl", (d.model_dump() for d in space.documents_))
        _dump(dest / "chunks.jsonl", (c.model_dump() for c in space.chunks))
        _dump(dest / "relations.jsonl", (r.model_dump(mode="json") for r in space.relations))
        (dest / "manifest.json").write_text(
            json.dumps(space.manifest.model_dump(), indent=2, sort_keys=True) + "\n"
        )


def _dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
