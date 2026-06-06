"""LLM protocol: text generation for enrichment (type, topics, tags, summaries).

Default: ``openai:gpt-5-mini`` (technical-spec §4.2). Names may carry a ``:model`` suffix,
which the registry splits and passes to the implementation as ``model=``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...
