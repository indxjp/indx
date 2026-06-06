"""Unit tests for the LangChain agent adapter — vendor SDK mocked, fully offline.

``langchain-core`` is absent here, so fake ``langchain_core.*`` modules are injected and the
dependency gate is neutralized. The fakes record exactly the fields the adapter sets so the
tool/retriever shapes can be asserted.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.agent.langchain  # noqa: F401  -- force-load before any require_extra patch
from indx.agent import KnowledgeConnector
from indx.errors import MissingExtraError


class _Document:
    def __init__(self, *, id: str, page_content: str, metadata: dict[str, Any]) -> None:  # noqa: A002
        self.id = id
        self.page_content = page_content
        self.metadata = metadata


class _StructuredTool:
    def __init__(self, *, func: Any, name: str, description: str) -> None:
        self.func = func
        self.name = name
        self.description = description

    @classmethod
    def from_function(cls, *, func: Any, name: str, description: str) -> _StructuredTool:
        return cls(func=func, name=name, description=description)


class _BaseRetriever:
    """Plain stand-in for langchain's pydantic ``BaseRetriever``."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _CallbackManagerForRetrieverRun:  # marker type only
    pass


@pytest.fixture
def fake_langchain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.agent.langchain.require_extra", lambda *a, **k: None)

    pkg = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")
    documents.Document = _Document  # type: ignore[attr-defined]
    tools = types.ModuleType("langchain_core.tools")
    tools.StructuredTool = _StructuredTool  # type: ignore[attr-defined]
    retrievers = types.ModuleType("langchain_core.retrievers")
    retrievers.BaseRetriever = _BaseRetriever  # type: ignore[attr-defined]
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.CallbackManagerForRetrieverRun = _CallbackManagerForRetrieverRun  # type: ignore[attr-defined]

    for name, module in {
        "langchain_core": pkg,
        "langchain_core.documents": documents,
        "langchain_core.tools": tools,
        "langchain_core.retrievers": retrievers,
        "langchain_core.callbacks": callbacks,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_langchain_tools_shape(fake_langchain: None, kb: KnowledgeConnector) -> None:
    tools = kb.langchain()
    assert [t.name for t in tools] == ["indx_search", "indx_overview", "indx_get_document"]
    # The search tool actually retrieves through the connector.
    result = tools[0].func(query="remote work", k=1)
    assert result["count"] >= 1
    assert result["hits"][0]["source"] == "people/remote-work.md"


def test_langchain_retriever_returns_documents(
    fake_langchain: None, kb: KnowledgeConnector
) -> None:
    retriever = kb.langchain_retriever(k=2)
    docs = retriever._get_relevant_documents("remote work from anywhere")
    assert docs
    top = docs[0]
    assert isinstance(top, _Document)
    assert top.page_content
    # Provenance folded into JSON-primitive metadata (matches the langchain writer).
    assert top.metadata["source"] == "people/remote-work.md"
    assert top.metadata["doc_type"] == "policy"
    assert isinstance(top.metadata["score"], float)


def test_langchain_without_extra_raises(kb: KnowledgeConnector) -> None:
    with pytest.raises(MissingExtraError):
        kb.langchain()
