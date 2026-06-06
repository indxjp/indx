"""Documented-API contract tests.

Executes the key snippets from the published docsite end-to-end on the OFFLINE CORE STACK
(parser=plaintext, llm=none, embedder=hash, store=jsonl, output=jsonl) over a tiny temp
directory. These assert the *documented* public surface actually imports and runs without
AttributeError/TypeError/ImportError — the acceptance bar for "documented API runs".
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_top_level_imports() -> None:
    """Every symbol the SDK docs import from ``indx`` is importable (sdk.md, data-models.md)."""
    from indx import (  # noqa: F401
        Chunk,
        DirectoryPipeline,
        Document,
        KnowledgeSpace,
        ParsedDoc,
        Relation,
        RelationType,
        SearchHit,
        Source,
        SpaceContext,
        SpaceStats,
    )


def test_subpackage_protocol_imports() -> None:
    """Sub-package protocol imports documented in the SDK surface resolve (technical-spec §8.1)."""
    from indx.embed import Embedder  # noqa: F401
    from indx.llm import LLM, VLM  # noqa: F401
    from indx.output import OutputWriter  # noqa: F401

    # Store / OutputWriter are documented aliases of the internal protocols.
    from indx.output.base import Writer
    from indx.parsers import Parser  # noqa: F401
    from indx.store import Store  # noqa: F401
    from indx.store.base import VectorStore

    assert Store is VectorStore
    assert OutputWriter is Writer


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny two-file source directory for the offline core stack."""
    src = tmp_path / "handbook"
    src.mkdir()
    (src / "policy.txt").write_text(
        "# Leave Policy\n\nEmployees may take paid leave.\n\nRequests go to the manager.",
        encoding="utf-8",
    )
    (src / "guide.txt").write_text(
        "# Onboarding Guide\n\nWelcome to the team and the company.",
        encoding="utf-8",
    )
    return src


def _offline_pipeline() -> object:
    from indx import DirectoryPipeline

    return DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="jsonl",
        output="jsonl",
    )


def test_pipeline_builds_and_runs(corpus: Path, tmp_path: Path) -> None:
    """``DirectoryPipeline(...).run(src, out)`` writes output and returns a KnowledgeSpace."""
    from indx import KnowledgeSpace

    out = tmp_path / "out.jsonl"
    pipe = _offline_pipeline()
    space = pipe.run(str(corpus), str(out))  # type: ignore[attr-defined]

    assert isinstance(space, KnowledgeSpace)
    assert out.exists()
    assert len(space.documents()) == 2


def test_fluent_stage_management(corpus: Path) -> None:
    """``.use()``, ``.drop()``, and ``.insert()`` are chainable and mutate the stage list."""
    from indx.core.context import SpaceContext

    class NoOpStage:
        name = "noop"

        def run(self, ctx: SpaceContext) -> SpaceContext:
            return ctx

    pipe = _offline_pipeline()
    # Component rebinding via use().
    returned = pipe.use(embedder="hash")  # type: ignore[attr-defined]
    assert returned is pipe

    pipe.drop("enrich").drop("embed-pack")  # type: ignore[attr-defined]
    names = [s.name for s in pipe.stages()]  # type: ignore[attr-defined]
    assert "enrich" not in names
    assert "embed-pack" not in names

    pipe.insert(0, NoOpStage())  # type: ignore[attr-defined]
    assert pipe.stages()[0].name == "noop"  # type: ignore[attr-defined]


def test_knowledge_space_accessors(corpus: Path, tmp_path: Path) -> None:
    """stats / documents(type=) / search(...) expose the documented shapes (sdk.md)."""
    from indx import SearchHit, SpaceStats

    out = tmp_path / "out.jsonl"
    space = _offline_pipeline().run(str(corpus), str(out))  # type: ignore[attr-defined]

    # stats is a SpaceStats with the documented counts.
    assert isinstance(space.stats, SpaceStats)
    assert space.stats.documents == 2
    assert space.stats.chunks >= 2

    # documents(type=...) filters; an unknown type yields nothing.
    a_type = space.documents()[0].doc_type
    assert all(d.doc_type == a_type for d in space.documents(type=a_type))
    assert space.documents(type="__no_such_type__") == []

    # search returns SearchHit objects exposing chunk / score / neighbors / source.
    hits = space.search("leave policy", k=3)
    assert hits
    for hit in hits:
        assert isinstance(hit, SearchHit)
        assert isinstance(hit.score, float)
        assert hit.chunk is not None
        assert isinstance(hit.neighbors, list)
        # .source is a shorthand for hit.chunk.source — accessible without AttributeError.
        assert hit.source == hit.chunk.source
        # ...and it is actually populated: docs cite hit.source.path/.folder/.type
        # (quickstart.mdx, sdk.md §search, inspect-and-query.mdx). The pack stage stamps it.
        assert hit.source is not None
        assert isinstance(hit.source.path, str) and hit.source.path


def test_save_load_round_trip(corpus: Path, tmp_path: Path) -> None:
    """``space.save(archive)`` then ``KnowledgeSpace.load(archive)`` round-trips (sdk.md)."""
    from indx import KnowledgeSpace

    out = tmp_path / "out.jsonl"
    space = _offline_pipeline().run(str(corpus), str(out))  # type: ignore[attr-defined]

    archive = tmp_path / "space.indx"
    space.save(str(archive))
    assert archive.exists()

    loaded = KnowledgeSpace.load(str(archive))
    assert isinstance(loaded, KnowledgeSpace)
    assert len(loaded.documents()) == len(space.documents())
    assert len(loaded.chunks) == len(space.chunks)


def test_byo_custom_parser_runs(corpus: Path, tmp_path: Path) -> None:
    """A BYO Parser instance (structurally satisfying the protocol) can be passed and runs."""
    from indx import DirectoryPipeline, KnowledgeSpace
    from indx.core.parsed import Block, ParsedDoc
    from indx.parsers import Parser

    class TaggingParser:
        name = "tagging"

        def parse(self, path: Path) -> ParsedDoc:
            return ParsedDoc(
                source_path=str(path),
                parser=self.name,
                blocks=[Block(kind="text", text="custom parsed body content", order=0)],
            )

    custom = TaggingParser()
    # Structurally satisfies the documented Parser protocol.
    assert isinstance(custom, Parser)

    out = tmp_path / "out.jsonl"
    pipe = DirectoryPipeline(
        parser=custom,
        llm="none",
        embedder="hash",
        store="jsonl",
        output="jsonl",
    )
    space = pipe.run(str(corpus), str(out))
    assert isinstance(space, KnowledgeSpace)
    assert space.chunks
    assert any("custom parsed body content" in c.text for c in space.chunks)
