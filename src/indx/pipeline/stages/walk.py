"""01 Walk — discover files, capture folder lineage, create Documents (FR-S01).

Optionally narrowed by a :class:`~indx.pipeline.filters.WalkFilter` (Feature 5): when a filter
is supplied, only files passing ALL its set predicates become Documents. The walk order and the
``max_files`` cap stay deterministic (iter_files yields a stable sorted order; the cap is the
first-N prefix of the post-filter, post-sort sequence).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.pipeline.filters import WalkFilter
from indx.utils.hashing import stable_hash
from indx.utils.io import iter_files


class WalkStage:
    name = "walk"

    def __init__(self, filter: WalkFilter | None = None) -> None:
        # ``None`` / empty filter == index everything (backward-compatible default).
        self._filter = filter

    def run(self, ctx: SpaceContext) -> SpaceContext:
        root = ctx.root
        flt = self._filter
        active = flt is not None and not flt.is_empty()
        kept = 0
        for path in iter_files(root):
            rel = path.relative_to(root)
            lineage = list(rel.parent.parts)
            stat_size = path.stat().st_size
            if active:
                assert flt is not None  # narrowed by ``active`` above
                if not flt.keep(
                    PurePosixPath(rel.as_posix()),
                    size_bytes=stat_size,
                    depth=len(lineage),
                ):
                    ctx.space.skipped_files_ += 1  # countable skip (report N indexed / M skipped)
                    continue
                if flt.max_files is not None and kept >= flt.max_files:
                    ctx.space.skipped_files_ += 1
                    continue
            ctx.space.documents_.append(
                Document(
                    id=stable_hash(str(rel)),
                    path=str(rel),
                    lineage=lineage,
                    size_bytes=stat_size,
                )
            )
            kept += 1
        return ctx
