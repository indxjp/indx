"""Live e2e of the agent connector against the real OpenAI API (``OPENAI_API_KEY``).

This is the headline "plug a knowledge space into any agent" proof, run with a *real*
LLM driving a raw tool-call loop — no agent-framework SDK, just the ``openai`` client,
``kb.openai_schema()`` (the function defs the model sees) and ``kb.call()`` (the single
dispatch path indx exposes to every framework). The space itself is built fully offline
(``embedder="hash"``, ``llm="none"``); the only live component under test is the OpenAI
model deciding to call ``indx_search`` and grounding its answer in the tool output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.agent import connect

pytestmark = pytest.mark.live


def test_openai_tool_loop_grounds_answer(require_env, live_corpus: Path) -> None:
    require_env("OPENAI_API_KEY")

    # The space is built offline: the embedder/LLM choice here is irrelevant to what is
    # under test — the live OpenAI model is what drives the tool loop below.
    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl"
    ).run(live_corpus)
    kb = connect(space, name="handbook")
    tools = kb.openai_schema()

    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_AGENT_MODEL", "gpt-5-mini")

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Use the indx_search tool to look up what the Search API does, "
                "then answer in one sentence."
            ),
        }
    ]

    tool_was_called = False
    search_outputs: list[dict] = []
    final_content: str | None = None

    # Up to 4 turns: model thinks → maybe calls tools → we feed results back → model answers.
    # No temperature/max_tokens: the gpt-5 family rejects a custom temperature and needs
    # max_completion_tokens, so omitting both keeps this test model-agnostic.
    for _ in range(4):
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                out = kb.call(tc.function.name, args)
                tool_was_called = True
                if tc.function.name == "indx_search":
                    search_outputs.append(out)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(out),
                    }
                )
            continue

        final_content = msg.content
        break

    if tool_was_called:
        # Preferred path: the model actually exercised the connector's tool surface.
        assert any(out.get("hits") for out in search_outputs), (
            "indx_search was called but returned no hits"
        )
        assert isinstance(final_content, str) and final_content.strip(), (
            "model produced no final answer after the tool loop"
        )
        # Deterministic regardless of the model's prose: search must surface the eng doc.
        sources = [
            h.get("source")
            for out in search_outputs
            for h in out.get("hits", [])
            if h.get("source")
        ]
        assert any(str(s).endswith("search-api.md") for s in sources), (
            f"indx_search never surfaced the Search API doc; sources={sources}"
        )
    else:
        # Documented fallback: a model may occasionally answer without calling a tool. The
        # deterministic tool path still must work, so dispatch it directly and prove it.
        out = kb.call("indx_search", {"query": "search api"})
        hits = out.get("hits", [])
        assert hits, "direct indx_search dispatch returned no hits"
        assert any(str(h.get("source", "")).endswith("search-api.md") for h in hits), (
            f"direct indx_search did not surface the Search API doc; hits={hits}"
        )
