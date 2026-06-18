"""Regression: CRUD reseal must recover ``references`` edges from the raw source root.

Issue #17 bug 2 made ``RelateStage._references`` read the *raw source file* for textual
documents, because the default (docling) parser strips link targets from the parsed text
(``[Beta](beta.md)`` -> bare word ``Beta``). But ``KnowledgeSpace._relate_full_corpus`` (the
CRUD ``add``/``update``/``remove`` reseal path) built its ``SpaceContext`` with ``root=Path(".")``,
so that raw read resolved against the cwd instead of ``manifest.source_root`` — and silently
lost every ``references`` edge a fresh build emits, breaking the "build then add seals
byte-identically to a single full build" contract.

These tests pin the relate context to ``manifest.source_root`` and exercise both the recovery
path (source present on disk) and graceful degradation (source gone).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.core.relation import RelationType

if TYPE_CHECKING:
    from pathlib import Path


def _space(source_root: str) -> KnowledgeSpace:
    """A 2-doc space whose chunk text has had link targets stripped (docling-style), so the
    ``references`` edge can only be recovered by reading the raw source under ``source_root``."""
    return KnowledgeSpace(
        manifest=Manifest(source_root=source_root),
        documents=[
            Document(id="d_alpha", path="alpha.md"),
            Document(id="d_beta", path="beta.md"),
        ],
        chunks=[
            # Link TARGET ``beta.md`` is gone — only the link TEXT ``Beta`` survives, exactly as
            # a normalizing parser leaves it. A chunk-text scan therefore finds no mention.
            Chunk(
                id="c_alpha",
                doc_id="d_alpha",
                position=0,
                text="Alpha is the entry point. See Beta.",
            ),
            Chunk(id="c_beta", doc_id="d_beta", position=0, text="Beta document body."),
        ],
    )


def _references(space: KnowledgeSpace) -> list[tuple[str, str]]:
    return [
        (r.src, r.dst) for r in space._relate_full_corpus() if r.type == RelationType.REFERENCES
    ]


def test_crud_reseal_recovers_references_from_source_root(tmp_path: Path) -> None:
    # Raw source retains the markdown link target the parser stripped from the chunk text.
    (tmp_path / "alpha.md").write_text(
        "Alpha is the entry point. See [Beta](beta.md).\n", encoding="utf-8"
    )
    (tmp_path / "beta.md").write_text("Beta document body.\n", encoding="utf-8")

    # With root pinned to source_root, the raw read resolves and the edge fires — matching a
    # fresh build. (Before the fix, root=Path(".") read the cwd, missed the file, and dropped it.)
    assert ("d_alpha", "d_beta") in _references(_space(str(tmp_path)))


def test_crud_reseal_degrades_gracefully_when_source_absent(tmp_path: Path) -> None:
    # source_root points at a dir that does not exist: the raw read fails and the scan falls
    # back to the (link-stripped) chunk text, so no edge fires — no crash, prior behavior kept.
    assert ("d_alpha", "d_beta") not in _references(_space(str(tmp_path / "absent")))
