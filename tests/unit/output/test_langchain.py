"""Unit tests for LangChainWriter — vendor SDK mocked, fully offline (coding-standards §11).

The ``langchain-core`` SDK is absent in this environment, so a fake
``langchain_core.documents`` module is injected into ``sys.modules`` and the dependency gate
is neutralized so construction proceeds. The fake ``Document`` records exactly the fields the
adapter sets (``id`` / ``page_content`` / ``metadata``) and exposes ``model_dump`` so JSONL
serialization round-trips.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace
from indx.errors import MissingExtraError
from indx.output.langchain import LangChainWriter


class _LCDocument:
    """Stand-in for ``langchain_core.documents.Document`` with a stable ``model_dump``."""

    def __init__(
        self,
        *,
        id: str,  # noqa: A002 - mirrors the LangChain field name exactly
        page_content: str,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.page_content = page_content
        self.metadata = metadata

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "page_content": self.page_content,
            "metadata": self.metadata,
        }


def _make_documents_module() -> types.ModuleType:
    module = types.ModuleType("langchain_core.documents")
    module.Document = _LCDocument  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_langchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``langchain_core*`` modules and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter binds require_extra by name (``from ... import``); neutralize that too.
    monkeypatch.setattr("indx.output.langchain.require_extra", lambda *a, **k: None)

    pkg = types.ModuleType("langchain_core")
    documents = _make_documents_module()
    monkeypatch.setitem(sys.modules, "langchain_core", pkg)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)


def _space() -> KnowledgeSpace:
    space = KnowledgeSpace()
    space.documents_.append(
        Document(
            id="d1",
            path="guide/intro.md",
            lineage=["guide"],
            doc_type="markdown",
            topics=["onboarding"],
            tags=["intro"],
        )
    )
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
            ),
            Chunk(id="c2", doc_id="d1", position=2, text="third", prev_id="c1"),
        ]
    )
    return space


def test_satisfies_writer_protocol(fake_langchain: None) -> None:
    from indx.output.base import Writer

    assert isinstance(LangChainWriter(), Writer)


def test_write_emits_one_document_per_chunk(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(_space(), tmp_path, name="handbook")

    out = tmp_path / "handbook.jsonl"
    assert out.exists()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert [r["id"] for r in rows] == ["c0", "c1", "c2"]
    # Chunk.text maps to page_content (not "text"/"content").
    assert [r["page_content"] for r in rows] == ["first", "second", "third"]


def test_document_carries_provenance_metadata(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(_space(), tmp_path)

    out = tmp_path / "handbook.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    middle = rows[1]
    assert middle["metadata"] == {
        "doc_id": "d1",
        "position": 1,
        "prev_id": "c0",
        "next_id": "c2",
        "source": "guide/intro.md",
        "lineage": ["guide"],
        "doc_type": "markdown",
        "topics": ["onboarding"],
        "tags": ["intro"],
    }
    # First chunk has no prev_id; the neighbor link is preserved as None.
    assert rows[0]["metadata"]["prev_id"] is None
    assert rows[0]["metadata"]["next_id"] == "c1"


def test_metadata_values_are_json_primitive(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(_space(), tmp_path)

    out = tmp_path / "handbook.jsonl"
    # If any metadata value were a non-JSON-serializable vendor object, json.loads
    # of the written line would have failed; reaching here proves primitives only.
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        for value in row["metadata"].values():
            assert value is None or isinstance(value, (str, int, float, list))


def test_jsonl_keys_are_sorted_for_determinism(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(_space(), tmp_path)

    line = (tmp_path / "handbook.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert list(row.keys()) == sorted(row.keys())


def test_dest_filename_follows_name(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(_space(), tmp_path, name="custom")
    assert (tmp_path / "custom.jsonl").exists()
    assert not (tmp_path / "handbook.jsonl").exists()


def test_empty_space_writes_empty_file(fake_langchain: None, tmp_path: Path) -> None:
    LangChainWriter().write(KnowledgeSpace(), tmp_path)
    out = tmp_path / "handbook.jsonl"
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        LangChainWriter()
