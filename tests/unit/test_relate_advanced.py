"""T1 — advanced Relate edges: ``references`` and ``duplicate-of`` (FR-S04, PRD risk R-3).

These are the high-precision, content-derived relations. The tests run the real offline stack
(Walk -> Parse -> Relate) so ``ctx.parsed`` is populated exactly as it is in production, then
assert on the emitted edges. Existing structural relations (sibling/parent/continues) are
covered by ``test_stages.py`` and the corpus suite; here we focus on the new behavior and on
its precision guarantees (no spurious edges).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.core.context import SpaceContext
from indx.core.knowledge_space import KnowledgeSpace
from indx.core.relation import Relation, RelationType
from indx.parsers.plaintext import PlainTextParser
from indx.pipeline.stages import ParseStage, RelateStage, WalkStage


def _build(root: Path) -> SpaceContext:
    """Walk -> Parse -> Relate over an on-disk tree, returning the populated context."""
    ctx = SpaceContext(root=root, space=KnowledgeSpace())
    ctx = WalkStage().run(ctx)
    ctx = ParseStage(PlainTextParser()).run(ctx)
    return RelateStage().run(ctx)


def _edges(ctx: SpaceContext, rel_type: RelationType) -> set[tuple[str, str]]:
    """Edges of a type as (src_path, dst_path) pairs for readable assertions."""
    path_of = {d.id: d.path for d in ctx.space.documents_}
    return {(path_of[r.src], path_of[r.dst]) for r in ctx.space.relations if r.type == rel_type}


# ─────────────────────────────────────────────────────────────────────────────
# references
# ─────────────────────────────────────────────────────────────────────────────


def test_references_resolves_markdown_link(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("See the policy in [retention](legal/retention.md).\n")
    (tmp_path / "legal").mkdir()
    (tmp_path / "legal" / "retention.md").write_text("Data is retained 90 days.\n")

    ctx = _build(tmp_path)
    assert ("guide.md", "legal/retention.md") in _edges(ctx, RelationType.REFERENCES)


def test_references_resolves_bare_filename(tmp_path: Path) -> None:
    (tmp_path / "overview.md").write_text("Read setup.md before you begin.\n")
    (tmp_path / "setup.md").write_text("Install the toolchain.\n")

    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert ("overview.md", "setup.md") in refs


def test_references_drops_ambiguous_basename(tmp_path: Path) -> None:
    # Two files share the basename README.md in different folders -> a bare "README.md" mention
    # is ambiguous and must NOT produce an edge (precision over recall, R-3).
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "README.md").write_text("Alpha section.\n")
    (tmp_path / "b" / "README.md").write_text("Beta section.\n")
    (tmp_path / "guide.md").write_text("Refer to README.md for details.\n")

    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert not any(src == "guide.md" for src, _ in refs)


def test_references_ignores_unknown_and_self_and_urls(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text(
        "Missing target nonexistent.md, self doc.md, and a link "
        "[site](https://example.com/page.md).\n"
    )
    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert all(src != "doc.md" for src, _ in refs)


def test_references_strips_anchor_fragment(tmp_path: Path) -> None:
    (tmp_path / "index.md").write_text("Jump to [setup](guide.md#install).\n")
    (tmp_path / "guide.md").write_text("Setup instructions.\n")
    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert ("index.md", "guide.md") in refs


def test_references_dedupes_repeated_mentions(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Use b.md. Again b.md. And once more b.md.\n")
    (tmp_path / "b.md").write_text("Body.\n")
    edges = [r for r in _build(tmp_path).space.relations if r.type == RelationType.REFERENCES]
    assert len(edges) == 1


# ─────────────────────────────────────────────────────────────────────────────
# duplicate-of
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_of_exact_copies(tmp_path: Path) -> None:
    (tmp_path / "original.txt").write_text("The quick brown fox.\n")
    (tmp_path / "copy.txt").write_text("The quick brown fox.\n")

    dups = _edges(_build(tmp_path), RelationType.DUPLICATE_OF)
    # Deterministic path-ordered direction: copy.txt < original.txt.
    assert ("copy.txt", "original.txt") in dups


def test_duplicate_of_ignores_whitespace_and_case(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello   World\n\nAgain\n")
    (tmp_path / "b.txt").write_text("hello world\nagain\n")
    dups = _edges(_build(tmp_path), RelationType.DUPLICATE_OF)
    assert ("a.txt", "b.txt") in dups


def test_duplicate_of_not_emitted_for_distinct_docs(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Completely different content here.\n")
    (tmp_path / "b.txt").write_text("Another unrelated paragraph entirely.\n")
    assert _edges(_build(tmp_path), RelationType.DUPLICATE_OF) == set()


def test_duplicate_of_empty_docs_not_duplicates(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("\n")
    (tmp_path / "b.txt").write_text("   \n")
    assert _edges(_build(tmp_path), RelationType.DUPLICATE_OF) == set()


def test_duplicate_of_chains_three_copies_deterministically(tmp_path: Path) -> None:
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text("identical body\n")
    dups = _edges(_build(tmp_path), RelationType.DUPLICATE_OF)
    # Consecutive chain in path order, not a full clique.
    assert dups == {("a.txt", "b.txt"), ("b.txt", "c.txt")}


# ─────────────────────────────────────────────────────────────────────────────
# coexistence / no regression of structural relations
# ─────────────────────────────────────────────────────────────────────────────


def test_advanced_relations_coexist_with_structural(tmp_path: Path) -> None:
    (tmp_path / "index.md").write_text("Top index. See guide.md.\n")
    (tmp_path / "guide.md").write_text("Guide body.\n")
    (tmp_path / "guide-copy.md").write_text("Guide body.\n")

    ctx = _build(tmp_path)
    types = {r.type for r in ctx.space.relations}
    assert RelationType.SIBLING in types
    assert RelationType.PARENT in types  # index.md is a parent of its folder-mates
    assert RelationType.REFERENCES in types
    assert RelationType.DUPLICATE_OF in types


def test_relate_without_parse_emits_no_content_edges(tmp_path: Path) -> None:
    # Run Relate directly after Walk (ctx.parsed is empty): structural edges still appear,
    # content-derived edges are simply absent rather than raising.
    (tmp_path / "a.md").write_text("See b.md.\n")
    (tmp_path / "b.md").write_text("See b.md.\n")
    ctx = SpaceContext(root=tmp_path, space=KnowledgeSpace())
    ctx = RelateStage().run(WalkStage().run(ctx))
    types = {r.type for r in ctx.space.relations}
    assert RelationType.REFERENCES not in types
    assert RelationType.DUPLICATE_OF not in types


def test_emitted_relations_are_relation_instances(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("Link to y.md here.\n")
    (tmp_path / "y.md").write_text("Y body.\n")
    rels = _build(tmp_path).space.relations
    assert rels and all(isinstance(r, Relation) for r in rels)


@pytest.mark.parametrize("ext", [".md", ".txt"])
def test_references_across_file_types(tmp_path: Path, ext: str) -> None:
    (tmp_path / f"src{ext}").write_text(f"points at dst{ext}\n")
    (tmp_path / f"dst{ext}").write_text("destination\n")
    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert (f"src{ext}", f"dst{ext}") in refs
