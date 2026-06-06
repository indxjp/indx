"""Unit tests for the Pydantic AI adapter — ``pydantic_ai`` package mocked, offline."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.agent.pydantic_ai  # noqa: F401  -- force-load before any require_extra patch
from indx.agent import KnowledgeConnector
from indx.errors import MissingExtraError


class _Tool:
    """Stand-in for ``pydantic_ai.Tool``: records the function plus name/description."""

    def __init__(self, function: Any, *, name: str, description: str) -> None:
        self.function = function
        self.name = name
        self.description = description


@pytest.fixture
def fake_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.agent.pydantic_ai.require_extra", lambda *a, **k: None)
    module = types.ModuleType("pydantic_ai")
    module.Tool = _Tool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydantic_ai", module)


def test_pydantic_ai_tools_shape(fake_pydantic_ai: None, kb: KnowledgeConnector) -> None:
    tools = kb.pydantic_ai()
    assert [t.name for t in tools] == ["indx_search", "indx_overview", "indx_get_document"]
    assert tools[0].description.startswith("Semantic search")
    # The wrapped function dispatches to the connector.
    assert tools[0].function(query="remote", k=1)["count"] >= 1
    assert tools[2].function(path_or_id="onboarding.md")["chunk_count"] == 1


def test_pydantic_ai_without_extra_raises(kb: KnowledgeConnector) -> None:
    with pytest.raises(MissingExtraError):
        kb.pydantic_ai()
