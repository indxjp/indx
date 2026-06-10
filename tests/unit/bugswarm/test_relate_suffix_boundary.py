"""Regression: relative-path mention resolution must respect path-component boundaries.

``_resolve_mention`` previously accepted a unique-basename match whenever the stored
normalized path ended with the mention string via plain ``str.endswith``. That suffix test
is not slash-delimited, so a mention ``sub/a.md`` wrongly resolved against a document stored
at ``xsub/a.md`` (because ``"xsub/a.md".endswith("sub/a.md")``), emitting a spurious
``references`` edge and violating the module's high-precision contract.
"""

from __future__ import annotations

from pathlib import Path

from indx.core.context import SpaceContext
from indx.core.knowledge_space import KnowledgeSpace
from indx.core.relation import RelationType
from indx.parsers.plaintext import PlainTextParser
from indx.pipeline.stages import ParseStage, RelateStage, WalkStage


def _build(root: Path) -> SpaceContext:
    ctx = SpaceContext(root=root, space=KnowledgeSpace())
    ctx = WalkStage().run(ctx)
    ctx = ParseStage(PlainTextParser()).run(ctx)
    return RelateStage().run(ctx)


def _edges(ctx: SpaceContext, rel_type: RelationType) -> set[tuple[str, str]]:
    path_of = {d.id: d.path for d in ctx.space.documents_}
    return {(path_of[r.src], path_of[r.dst]) for r in ctx.space.relations if r.type == rel_type}


def test_relative_mention_does_not_cross_path_boundary(tmp_path: Path) -> None:
    # The only document with basename a.md lives at xsub/a.md. A mention of sub/a.md (where
    # sub/ does not exist) must NOT resolve to it: "sub" is not a real parent segment of
    # "xsub", it is only a non-slash-delimited suffix.
    (tmp_path / "xsub").mkdir()
    (tmp_path / "xsub" / "a.md").write_text("Body of a.\n")
    (tmp_path / "guide.md").write_text("Refer to [a](sub/a.md) for details.\n")

    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert ("guide.md", "xsub/a.md") not in refs
    assert not any(src == "guide.md" for src, _ in refs)


def test_relative_mention_still_resolves_on_true_boundary(tmp_path: Path) -> None:
    # A partial relative mention (sub/a.md) that is a true slash-delimited suffix of the
    # stored path (deeper/sub/a.md) must still resolve through the suffix branch.
    (tmp_path / "deeper").mkdir()
    (tmp_path / "deeper" / "sub").mkdir()
    (tmp_path / "deeper" / "sub" / "a.md").write_text("Body of a.\n")
    (tmp_path / "guide.md").write_text("Refer to [a](sub/a.md) for details.\n")

    refs = _edges(_build(tmp_path), RelationType.REFERENCES)
    assert ("guide.md", "deeper/sub/a.md") in refs
