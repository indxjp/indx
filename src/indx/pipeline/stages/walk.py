"""01 Walk — discover files, capture folder lineage, create Documents (FR-S01)."""

from __future__ import annotations

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.utils.hashing import stable_hash
from indx.utils.io import iter_files


class WalkStage:
    name = "walk"

    def run(self, ctx: SpaceContext) -> SpaceContext:
        root = ctx.root
        for path in iter_files(root):
            rel = path.relative_to(root)
            ctx.space.documents_.append(
                Document(
                    id=stable_hash(str(rel)),
                    path=str(rel),
                    lineage=list(rel.parent.parts),
                    size_bytes=path.stat().st_size,
                )
            )
        return ctx
