"""Extras e2e for the agent framework ADAPTERS, against each vendor's real library.

The core ``indx.agent.connect`` connector (search / overview / get_document) needs no
extra, but the framework bridges do — LangChain (``langchain_core``), Pydantic AI
(``pydantic_ai``), the OpenAI Agents SDK (``agents``), and the Claude Agent SDK
(``claude_agent_sdk``). Each test self-skips via ``requires_lib`` when its library isn't
importable, so a matrix leg that installed only one extra stays green.

All tests share a single offline, deterministic (``seed=0``, ``hash`` embedder) space
built from the ``acme_kb`` corpus and wrapped with ``connect``. Assertions target the
contractually stable surface (the three tool names, documented call signatures, JSON-able
results) rather than version-fragile vendor internals.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.agent import KnowledgeConnector, connect

pytestmark = pytest.mark.extras

_CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "acme_kb"
_TOOL_NAMES = {"indx_search", "indx_overview", "indx_get_document"}


@pytest.fixture(scope="module")
def kb() -> KnowledgeConnector:
    """An offline space over the acme_kb corpus, wrapped as a KnowledgeConnector."""
    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl"
    ).run(_CORPUS)
    return connect(space, name="acme")


def test_langchain_tools_and_retriever(
    requires_lib: Callable[[str], None], kb: KnowledgeConnector
) -> None:
    requires_lib("langchain_core")

    tools = kb.langchain()
    assert len(tools) == 3
    assert {tool.name for tool in tools} == _TOOL_NAMES

    search_tool = next(tool for tool in tools if tool.name == "indx_search")
    result = search_tool.invoke({"query": "incident severity", "k": 1})
    assert result, "search tool returned an empty result"
    assert "incident-response" in str(result)

    retriever = kb.langchain_retriever(k=2)
    docs = retriever.invoke("incident on-call severity")
    assert docs, "retriever returned no documents"
    for doc in docs:
        assert isinstance(doc.page_content, str)
        assert isinstance(doc.metadata, dict)


def test_pydantic_ai_tools(requires_lib: Callable[[str], None], kb: KnowledgeConnector) -> None:
    requires_lib("pydantic_ai")

    tools = kb.pydantic_ai()
    assert len(tools) == 3

    def _ident(tool: object) -> str:
        name = getattr(tool, "name", "") or ""
        fn_name = getattr(getattr(tool, "function", None), "__name__", "") or ""
        return f"{name} {fn_name}"

    search_tool = next(tool for tool in tools if "search" in _ident(tool))
    fn = getattr(search_tool, "function", None)
    assert callable(fn), "pydantic_ai search tool has no callable .function"

    result = fn(query="retirement 401k", k=1)
    assert "hits" in result
    assert result["hits"][0]["source"].endswith("retirement-401k.md")


def test_openai_agent_tools(requires_lib: Callable[[str], None], kb: KnowledgeConnector) -> None:
    requires_lib("agents")

    tools = kb.openai()
    assert len(tools) == 3
    assert {tool.name for tool in tools} == _TOOL_NAMES
    assert all(isinstance(tool.params_json_schema, dict) for tool in tools)


def test_claude_agent_server(requires_lib: Callable[[str], None], kb: KnowledgeConnector) -> None:
    requires_lib("claude_agent_sdk")

    server = kb.claude(name="acme")
    assert server is not None

    # The SDK server object's shape is version-dependent; assert defensively when it
    # exposes a name or tool listing, but building without raising is the contract.
    try:
        name = getattr(server, "name", None)
        if name is not None:
            assert "acme" in str(name)
        tools = getattr(server, "tools", None)
        if tools is not None and hasattr(tools, "__len__"):
            assert len(tools) == 3
    except AssertionError:
        raise
    except Exception:  # noqa: BLE001 - lenient: object shape varies across SDK versions
        pass
