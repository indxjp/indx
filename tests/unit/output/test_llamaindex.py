"""Unit tests for LlamaIndexWriter — vendor SDK mocked, fully offline (coding-standards §11).

The ``llama-index-core`` SDK is absent in this environment, so a fake
``llama_index.core.schema`` module is injected into ``sys.modules`` and the dependency gate
is neutralized so construction proceeds. The fake ``TextNode`` records exactly the fields
the adapter sets and exposes ``model_dump`` so JSONL serialization round-trips.
"""

from __future__ import annotations

import json
import sys
import types
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace
from indx.errors import MissingExtraError
from indx.output.llamaindex import LlamaIndexWriter


class _NodeRelationship(StrEnum):
    """Stand-in for ``llama_index.core.schema.NodeRelationship``."""

    SOURCE = "1"
    PREVIOUS = "2"
    NEXT = "3"


class _RelatedNodeInfo:
    """Stand-in for ``llama_index.core.schema.RelatedNodeInfo``."""

    def __init__(self, *, node_id: str) -> None:
        self.node_id = node_id


class _TextNode:
    """Stand-in for ``llama_index.core.schema.TextNode`` with a stable ``model_dump``."""

    def __init__(
        self,
        *,
        id_: str,
        text: str,
        embedding: list[float] | None,
        metadata: dict[str, Any],
        relationships: dict[_NodeRelationship, _RelatedNodeInfo],
    ) -> None:
        self.id_ = id_
        self.text = text
        self.embedding = embedding
        self.metadata = metadata
        self.relationships = relationships

    def model_dump(self) -> dict[str, Any]:
        return {
            "id_": self.id_,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "relationships": {rel.value: info.node_id for rel, info in self.relationships.items()},
        }


def _make_schema_module() -> types.ModuleType:
    module = types.ModuleType("llama_index.core.schema")
    module.TextNode = _TextNode  # type: ignore[attr-defined]
    module.RelatedNodeInfo = _RelatedNodeInfo  # type: ignore[attr-defined]
    module.NodeRelationship = _NodeRelationship  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_llamaindex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``llama_index*`` modules and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter binds require_extra by name (``from ... import``); neutralize that too.
    monkeypatch.setattr("indx.output.llamaindex.require_extra", lambda *a, **k: None)

    pkg = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    schema = _make_schema_module()
    monkeypatch.setitem(sys.modules, "llama_index", pkg)
    monkeypatch.setitem(sys.modules, "llama_index.core", core)
    monkeypatch.setitem(sys.modules, "llama_index.core.schema", schema)


def _space() -> KnowledgeSpace:
    space = KnowledgeSpace()
    space.documents_.append(Document(id="d1", path="guide/intro.md"))
    space.chunks.extend(
        [
            Chunk(id="c0", doc_id="d1", position=0, text="first", next_id="c1"),
            Chunk(
                id="c1",
                doc_id="d1",
                position=1,
                text="second",
                prev_id="c0",
                next_id="c2",
                embedding=[0.1, 0.2],
            ),
            Chunk(id="c2", doc_id="d1", position=2, text="third", prev_id="c1"),
        ]
    )
    return space


def test_satisfies_writer_protocol(fake_llamaindex: None) -> None:
    from indx.output.base import Writer

    assert isinstance(LlamaIndexWriter(), Writer)


def test_write_emits_one_node_per_chunk(fake_llamaindex: None, tmp_path: Path) -> None:
    LlamaIndexWriter().write(_space(), tmp_path, name="handbook")

    out = tmp_path / "handbook.jsonl"
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert [r["id_"] for r in rows] == ["c0", "c1", "c2"]
    assert [r["text"] for r in rows] == ["first", "second", "third"]


def test_node_carries_metadata_and_relationships(fake_llamaindex: None, tmp_path: Path) -> None:
    LlamaIndexWriter().write(_space(), tmp_path)

    out = tmp_path / "handbook.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    middle = rows[1]
    assert middle["metadata"] == {
        "doc_id": "d1",
        "position": 1,
        "prev_id": "c0",
        "next_id": "c2",
        "source": "guide/intro.md",
    }
    assert middle["embedding"] == [0.1, 0.2]
    # SOURCE always present; PREVIOUS/NEXT present because this chunk has both neighbors.
    assert middle["relationships"] == {
        _NodeRelationship.SOURCE.value: "d1",
        _NodeRelationship.PREVIOUS.value: "c0",
        _NodeRelationship.NEXT.value: "c2",
    }
    # First chunk has no prev_id -> no PREVIOUS relationship.
    assert _NodeRelationship.PREVIOUS.value not in rows[0]["relationships"]
    assert rows[0]["relationships"][_NodeRelationship.NEXT.value] == "c1"


def test_dest_filename_follows_name(fake_llamaindex: None, tmp_path: Path) -> None:
    LlamaIndexWriter().write(_space(), tmp_path, name="custom")
    assert (tmp_path / "custom.jsonl").exists()
    assert not (tmp_path / "handbook.jsonl").exists()


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        LlamaIndexWriter()
