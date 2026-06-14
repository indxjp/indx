"""01 Walk — discover files, capture folder lineage, create Documents (FR-S01).

Optionally narrowed by a :class:`~indx.pipeline.filters.WalkFilter` (Feature 5): when a filter
is supplied, only files passing ALL its set predicates become Documents. The walk order and the
``max_files`` cap stay deterministic (iter_files yields a stable sorted order; the cap is the
first-N prefix of the post-filter, post-sort sequence).
"""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath

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
        # When ``max_files`` is set, the cap must select the first N in a GLOBAL root-relative
        # posix-path sort (the documented "first-N after the walk's stable sort"). iter_files'
        # per-directory DFS order is NOT that global sort (a root file ``sub.md`` sorts before
        # ``sub/x.md`` because '.' < '/', yet DFS descends into ``sub/`` first), so we materialize
        # the kept candidates and sort by their root-relative posix path before truncating.
        capped = active and flt.max_files is not None  # type: ignore[union-attr]
        candidates: list[tuple[str, PurePath, list[str], int]] = []  # (sort_key, rel, lineage, sz)
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
            if capped:
                candidates.append((rel.as_posix(), rel, lineage, stat_size))
                continue
            ctx.space.documents_.append(
                Document(
                    id=stable_hash(str(rel)),
                    path=str(rel),
                    lineage=lineage,
                    size_bytes=stat_size,
                )
            )
        if capped:
            assert flt is not None and flt.max_files is not None
            candidates.sort(key=lambda c: c[0])  # global root-relative posix-path sort
            for _key, crel, lineage, stat_size in candidates[: flt.max_files]:
                ctx.space.documents_.append(
                    Document(
                        id=stable_hash(str(crel)),
                        path=str(crel),
                        lineage=lineage,
                        size_bytes=stat_size,
                    )
                )
            ctx.space.skipped_files_ += max(0, len(candidates) - flt.max_files)
        return ctx
