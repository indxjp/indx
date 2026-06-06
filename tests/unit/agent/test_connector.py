"""Unit tests for the framework-agnostic KnowledgeConnector — fully offline, no vendor SDKs.

Exercises the documented operations (search / overview / get_document), the raw tool-spec
emitters, and the ``call`` dispatcher on the in-core ``hash`` retrieval path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indx.agent import (
    GET_DOCUMENT_TOOL,
    OVERVIEW_TOOL,
    SEARCH_TOOL,
    DocumentDetail,
    KnowledgeConnector,
    SearchResults,
    SpaceOverview,
    connect,
)
from indx.core.knowledge_space import KnowledgeSpace


def test_search_returns_flat_hits(kb: KnowledgeConnector) -> None:
    results = kb.search("remote work from anywhere", k=2)
    assert isinstance(results, SearchResults)
    assert results.query == "remote work from anywhere"
    assert 1 <= results.count <= 2
    top = results.hits[0]
    # Provenance an agent can cite.
    assert top.source == "people/remote-work.md"
    assert top.folder == "people"
    assert top.doc_type == "policy"
    assert top.topics == ["remote", "work"]
    assert isinstance(top.score, float)
    assert top.text


def test_search_is_json_primitive(kb: KnowledgeConnector) -> None:
    payload = kb.search("onboarding", k=3).model_dump(mode="json")
    # Round-trips through json => no nested vendor objects leaked in.
    assert json.loads(json.dumps(payload))["count"] >= 1


def test_search_doc_type_filter(kb: KnowledgeConnector) -> None:
    results = kb.search("anything", k=5, doc_type="guide")
    assert results.count >= 1
    assert {h.doc_type for h in results.hits} == {"guide"}


def test_search_with_context_includes_neighbors(space: KnowledgeSpace) -> None:
    kb = KnowledgeConnector(space, with_context=True)
    results = kb.search("remote-first employees", k=1)
    # c0's neighbor is c1; context carries the adjacent chunk text.
    assert results.hits[0].context


def test_overview_summarizes_space(kb: KnowledgeConnector) -> None:
    ov = kb.overview(sample=10)
    assert isinstance(ov, SpaceOverview)
    assert ov.name == "handbook"
    assert ov.documents == 2
    assert ov.chunks == 3
    assert ov.embedding_model == "hash"
    assert ov.types == {"policy": 1, "guide": 1}
    assert {d.path for d in ov.sample_documents} == {
        "people/remote-work.md",
        "guide/onboarding.md",
    }
    # Cards carry the enrichment an agent uses to decide relevance.
    policy = next(d for d in ov.sample_documents if d.doc_type == "policy")
    assert policy.summary == "How remote work works at Acme."


def test_overview_sample_zero_lists_no_documents(kb: KnowledgeConnector) -> None:
    assert kb.overview(sample=0).sample_documents == []


def test_get_document_by_path(kb: KnowledgeConnector) -> None:
    detail = kb.get_document("people/remote-work.md")
    assert isinstance(detail, DocumentDetail)
    assert detail.chunk_count == 2
    # Full text joins the document's chunks in position order.
    assert detail.text.startswith("Acme is remote-first")
    assert "Core collaboration hours" in detail.text


def test_get_document_by_id_and_suffix(kb: KnowledgeConnector) -> None:
    assert kb.get_document("d2").path == "guide/onboarding.md"
    # Suffix match: a bare filename resolves to the full path.
    assert kb.get_document("onboarding.md").id == "d2"


def test_get_document_missing_returns_none(kb: KnowledgeConnector) -> None:
    assert kb.get_document("nope/missing.md") is None


def test_tool_specs_shapes(kb: KnowledgeConnector) -> None:
    names = [t.name for t in kb.tools()]
    assert names == [SEARCH_TOOL.name, OVERVIEW_TOOL.name, GET_DOCUMENT_TOOL.name]

    openai = kb.openai_schema()
    assert all(t["type"] == "function" for t in openai)
    assert openai[0]["function"]["name"] == "indx_search"
    assert openai[0]["function"]["parameters"]["required"] == ["query"]

    anthropic = kb.anthropic_schema()
    assert anthropic[0]["name"] == "indx_search"
    assert "input_schema" in anthropic[0]


def test_call_dispatch(kb: KnowledgeConnector) -> None:
    assert kb.call("indx_search", {"query": "remote", "k": 1})["count"] >= 1
    assert kb.call("indx_overview", {})["documents"] == 2
    assert kb.call("indx_get_document", {"path_or_id": "d1"})["chunk_count"] == 2
    # A miss returns an error dict, not an exception.
    assert "error" in kb.call("indx_get_document", {"path_or_id": "ghost"})


def test_call_unknown_tool_raises(kb: KnowledgeConnector) -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        kb.call("indx_delete_everything", {})


def test_connect_from_space_instance(space: KnowledgeSpace) -> None:
    kb = connect(space, name="custom")
    assert isinstance(kb, KnowledgeConnector)
    assert kb.name == "custom"
    assert kb.overview().documents == 2


def test_open_from_archive_path(space: KnowledgeSpace, tmp_path: Path) -> None:
    archive = tmp_path / "kb.indx"
    space.save(str(archive))
    kb = connect(archive)
    # Name defaults to the archive stem — the "USB drive" label.
    assert kb.name == "kb"
    assert kb.search("onboarding", k=1).count >= 1
