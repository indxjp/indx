"""Live e2e against the real Anthropic API (``ANTHROPIC_API_KEY``).

Anthropic is an LLM backend only. The scaffold Enrich stage is LLM-free, so a build
never calls the messages API; we drive the ``anthropic`` adapter directly through the
public registry (the same object the pipeline would construct) to prove the real
completion round-trip works end to end.
"""

from __future__ import annotations

import pytest

from indx.registry import get_llm

pytestmark = pytest.mark.live


def test_anthropic_llm_completion(require_env) -> None:
    require_env("ANTHROPIC_API_KEY")

    llm = get_llm("anthropic")  # default model: claude-haiku-4-5
    reply = llm.complete("Reply with exactly one word: pong", max_tokens=16)

    assert isinstance(reply, str)
    assert reply.strip(), "live Anthropic completion returned empty text"


def test_anthropic_respects_system_prompt(require_env) -> None:
    require_env("ANTHROPIC_API_KEY")

    llm = get_llm("anthropic")
    reply = llm.complete(
        "What is two plus two?",
        system="You answer with digits only, no words.",
        max_tokens=16,
    )

    assert "4" in reply, f"expected the digit 4 in the reply, got: {reply!r}"
