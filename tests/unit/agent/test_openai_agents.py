"""Unit tests for the OpenAI Agents SDK adapter — ``agents`` package mocked, offline."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.agent.openai_agents  # noqa: F401  -- force-load before any require_extra patch
from indx.agent import KnowledgeConnector
from indx.errors import MissingExtraError


class _FunctionTool:
    """Stand-in for ``agents.FunctionTool``: keeps the wrapped function callable by name."""

    def __init__(self, fn: Any) -> None:
        self.name = fn.__name__
        self.description = (fn.__doc__ or "").strip()
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.agent.openai_agents.require_extra", lambda *a, **k: None)
    module = types.ModuleType("agents")
    module.function_tool = _FunctionTool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", module)


def test_openai_agent_tools_shape(fake_agents: None, kb: KnowledgeConnector) -> None:
    tools = kb.openai()
    assert [t.name for t in tools] == ["indx_search", "indx_overview", "indx_get_document"]
    # Each tool carries a description from its docstring and dispatches to the connector.
    assert tools[0].description
    assert tools[0](query="onboarding", k=2)["count"] >= 1
    assert tools[1]()["documents"] == 2
    assert tools[2](path_or_id="d1")["chunk_count"] == 2


def test_openai_without_extra_raises(kb: KnowledgeConnector) -> None:
    with pytest.raises(MissingExtraError):
        kb.openai()
