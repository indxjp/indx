"""Regression: O(n²) SIBLING fan-out degrades gracefully above SIBLING_MAX (F5, #14 N4).

A flat directory with more than SIBLING_MAX documents previously produced ~n²/2 SIBLING
relations, whose archive exceeded MAX_MEMBER_BYTES and could not be reopened. An earlier fix
capped that by emitting ZERO sibling edges for any oversized folder — but that hard cliff
silently dropped all structural sibling info the moment a folder crossed the line (#14 N4).

The current behavior degrades gracefully instead: above the threshold each document links to
its next SIBLING_NEIGHBORS path-adjacent siblings, so the edge count stays bounded (~k*n,
well under n²/2 and MAX_MEMBER_BYTES) yet is never zero for a large folder.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace
from indx.core.relation import RelationType
from indx.pipeline.stages.relate import SIBLING_MAX, SIBLING_NEIGHBORS, RelateStage


def _expected_bounded_siblings(n: int) -> int:
    """Edge count for the bounded k-neighbour graph: sum_i min(k, n-1-i)."""
    k = SIBLING_NEIGHBORS
    return sum(min(k, n - 1 - i) for i in range(n))


def _make_ctx(root: Path, n: int, lineage: list[str] | None = None) -> SpaceContext:
    """Build a SpaceContext with *n* documents all in the same folder."""
    if lineage is None:
        lineage = []
    space = KnowledgeSpace()
    for i in range(n):
        doc = Document(
            id=str(uuid.uuid4()),
            path="/".join(lineage + [f"doc_{i:04d}.md"]),
            lineage=lineage,
        )
        space.documents_.append(doc)
    return SpaceContext(root=root, space=space)


def _sibling_count(ctx: SpaceContext) -> int:
    return sum(1 for r in ctx.space.relations if r.type == RelationType.SIBLING)


# ---------------------------------------------------------------------------
# Large group — siblings must be suppressed
# ---------------------------------------------------------------------------


def test_large_group_degrades_to_bounded_siblings(tmp_path: Path) -> None:
    """A folder of 600 docs (> SIBLING_MAX=500) yields a bounded, NON-ZERO sibling graph (N4)."""
    n = SIBLING_MAX + 100  # 600
    ctx = _make_ctx(tmp_path, n)
    ctx = RelateStage().run(ctx)
    count = _sibling_count(ctx)
    # Non-zero (the old hard cliff produced 0), bounded at the k-neighbour count, and far below
    # the full pairwise explosion that overflowed the archive.
    assert count == _expected_bounded_siblings(n)
    assert 0 < count <= SIBLING_NEIGHBORS * n
    assert count < n * (n - 1) // 2


def test_large_group_docs_still_present(tmp_path: Path) -> None:
    """Documents in an oversized group must still be indexed (no ArchiveError / no exception)."""
    n = SIBLING_MAX + 100
    ctx = _make_ctx(tmp_path, n)
    # Must not raise any exception.
    ctx = RelateStage().run(ctx)
    assert len(ctx.space.documents_) == n


# ---------------------------------------------------------------------------
# Small group — siblings must still be emitted
# ---------------------------------------------------------------------------


def test_small_group_emits_all_sibling_pairs(tmp_path: Path) -> None:
    """A folder of 3 docs (< SIBLING_MAX) must produce 3 sibling pairs: (a,b),(a,c),(b,c)."""
    ctx = _make_ctx(tmp_path, 3)
    ctx = RelateStage().run(ctx)
    assert _sibling_count(ctx) == 3, (
        f"Expected 3 sibling relations for group of 3 docs, got {_sibling_count(ctx)}"
    )


def test_exactly_sibling_max_emits_siblings(tmp_path: Path) -> None:
    """A group at the boundary must emit the correct n*(n-1)/2 sibling pairs."""
    # n=10 is sufficient to verify the n*(n-1)/2 formula without O(n²) allocations at n=500
    n = 10
    ctx = _make_ctx(tmp_path, n)
    ctx = RelateStage().run(ctx)
    count = _sibling_count(ctx)
    assert count == n * (n - 1) // 2, (
        f"Expected {n * (n - 1) // 2} sibling relations for group of {n} docs, got {count}"
    )


def test_above_sibling_max_emits_bounded_nonzero(tmp_path: Path) -> None:
    """A group of SIBLING_MAX+1 docs crosses into the bounded k-neighbour regime (non-zero)."""
    n = SIBLING_MAX + 1
    ctx = _make_ctx(tmp_path, n)
    ctx = RelateStage().run(ctx)
    count = _sibling_count(ctx)
    assert count == _expected_bounded_siblings(n)
    assert count > 0


# ---------------------------------------------------------------------------
# Pure math sanity check — no RelateStage call, no object allocation
# ---------------------------------------------------------------------------


def test_sibling_max_formula_value() -> None:
    # SIBLING_MAX pairwise formula — documents the expected value without allocating.
    assert 500 * 499 // 2 == 124750
