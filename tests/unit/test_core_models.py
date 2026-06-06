"""T1 — core domain models: construction, defaults, serialization."""

from __future__ import annotations

from indx.core import Chunk, Document, KnowledgeSpace, ParsedDoc, Relation, RelationType
from indx.core.parsed import Block


def test_document_lineage_defaults() -> None:
    doc = Document(id="d1", path="a/b/c.md", lineage=["a", "b"])
    assert doc.topics == [] and doc.summary is None


def test_parseddoc_text_orders_blocks() -> None:
    parsed = ParsedDoc(
        source_path="x",
        blocks=[Block(text="second", order=1), Block(text="first", order=0)],
    )
    assert parsed.text == "first\n\nsecond"


def test_relation_type_is_enum_value() -> None:
    rel = Relation(src="a", dst="b", type=RelationType.CONTINUES)
    assert rel.model_dump(mode="json")["type"] == "continues"


def test_knowledge_space_lookup_helpers() -> None:
    space = KnowledgeSpace(
        documents=[Document(id="d1", path="a.md")],
        chunks=[Chunk(id="c1", doc_id="d1", position=0, text="hi")],
    )
    assert space.document("d1").path == "a.md"
    assert space.document("missing") is None
    assert [c.id for c in space.chunks_for("d1")] == ["c1"]
