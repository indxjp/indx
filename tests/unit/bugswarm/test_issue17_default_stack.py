"""Issue #17 — three default-stack bugs.

Covers the two fixes that have an observable, parser-independent behavior:

* bug 2 — ``references`` must fire even when the parser normalized link targets out of the
  chunk text (the default docling parser does), by scanning the raw source on disk;
* bug 3 — the expanded ``out/index.json`` must carry the same per-doc adjacency
  (``chunk_ids``/``references``/``referenced_by``) the sealed ``.indx`` archive does.

Bug 1 (docling non-success status) is covered in ``tests/unit/parsers/test_docling.py``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from indx.archive import format as fmt
from indx.core import Chunk, Document, KnowledgeSpace
from indx.core.context import SpaceContext
from indx.core.knowledge_space import Manifest
from indx.core.parsed import Block, ParsedDoc
from indx.core.relation import Relation, RelationType
from indx.output.indx_writer import IndxWriter
from indx.pipeline.stages.relate import RelateStage

# --- Bug 2 -----------------------------------------------------------------------------------


def test_references_fire_when_parser_strips_link_targets(tmp_path: Path) -> None:
    # On disk, alpha.md links to beta.md. Simulate the default (docling) parser, which flattens
    # ``[Beta](beta.md)`` to the bare word "Beta": the parsed/chunk text carries NO link syntax.
    (tmp_path / "alpha.md").write_text(
        "Alpha is the entry point. See [Beta](beta.md) and [Gamma](gamma.md).\n"
    )
    (tmp_path / "beta.md").write_text("Beta details.\n")

    alpha = Document(id="alpha", path="alpha.md")
    beta = Document(id="beta", path="beta.md")
    space = KnowledgeSpace(documents=[alpha, beta])
    ctx = SpaceContext(root=tmp_path, space=space)
    # Parser-normalized text with link targets stripped — what docling actually produces.
    ctx.parsed["alpha"] = ParsedDoc(
        source_path="alpha.md",
        blocks=[Block(text="Alpha is the entry point. See Beta and Gamma.")],
    )
    ctx.parsed["beta"] = ParsedDoc(source_path="beta.md", blocks=[Block(text="Beta details.")])

    RelateStage().run(ctx)

    refs = [r for r in space.relations if r.type == RelationType.REFERENCES]
    # gamma.md is not a walked document, so it is dropped; beta resolves uniquely.
    assert [(r.src, r.dst) for r in refs] == [("alpha", "beta")]


def test_references_fall_back_to_parsed_text_when_source_unreadable(tmp_path: Path) -> None:
    # No files on disk: the textual-ext raw read fails and falls back to parsed text. With link
    # targets stripped there, no reference can resolve — and the stage must not raise on the
    # missing source files.
    alpha = Document(id="alpha", path="alpha.md")
    beta = Document(id="beta", path="beta.md")
    space = KnowledgeSpace(documents=[alpha, beta])
    ctx = SpaceContext(root=tmp_path, space=space)
    ctx.parsed["alpha"] = ParsedDoc(source_path="alpha.md", blocks=[Block(text="Alpha. See Beta.")])
    ctx.parsed["beta"] = ParsedDoc(source_path="beta.md", blocks=[Block(text="Beta.")])

    RelateStage().run(ctx)

    assert [r for r in space.relations if r.type == RelationType.REFERENCES] == []


# --- Bug 3 -----------------------------------------------------------------------------------


def _archive_documents(indx_path: Path) -> list[dict]:
    with zipfile.ZipFile(indx_path) as zf:
        body = zf.read(fmt.DOCUMENTS).decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line]


def test_index_json_matches_archive_adjacency(tmp_path: Path) -> None:
    # A build with ≥1 chunk per doc and a REFERENCES relation: the expanded index.json must carry
    # the same backfilled chunk_ids / references / referenced_by the sealed archive does.
    docs = [Document(id="alpha", path="alpha.md"), Document(id="beta", path="beta.md")]
    chunks = [
        Chunk(id="c0", doc_id="alpha", position=0, text="alpha text"),
        Chunk(id="c1", doc_id="beta", position=0, text="beta text"),
    ]
    space = KnowledgeSpace(
        documents=docs,
        chunks=chunks,
        manifest=Manifest(embedding_model="hash", embedding_dim=2, components={"store": "jsonl"}),
    )
    space.relations.append(Relation(src="alpha", dst="beta", type=RelationType.REFERENCES))

    IndxWriter().write(space, tmp_path, name="handbook")

    index = json.loads((tmp_path / "index.json").read_text())
    index_docs = index["documents"]
    archive_docs = _archive_documents(tmp_path / "handbook.indx")

    # The two representations agree document-for-document.
    assert index_docs == archive_docs

    by_id = {d["id"]: d for d in index_docs}
    # Adjacency is populated, not shipped empty.
    assert by_id["alpha"]["chunk_ids"] == ["c0"]
    assert by_id["beta"]["chunk_ids"] == ["c1"]
    assert [r["dst"] for r in by_id["alpha"]["references"]] == ["beta"]
    assert [r["src"] for r in by_id["beta"]["referenced_by"]] == ["alpha"]
