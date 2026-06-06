"""NullLLM — the ``none`` LLM: enrichment disabled. Ships in core, zero-dependency.

Selected by ``--llm none`` (or ``[enrich] llm = "none"``) to skip LLM enrichment entirely
while still producing a valid space (the deterministic scaffold enrichment still runs).
"""

from __future__ import annotations


class NullLLM:
    name = "none"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        return ""
