#!/usr/bin/env python3
"""Plug a knowledge space into an AI agent — fully offline, no agent SDK required.

This script builds a knowledge space with the zero-dependency core stack, then wraps it in a
:class:`~indx.agent.connector.KnowledgeConnector` (``connect``). It exercises the
framework-agnostic surface that needs *no* extras:

* ``kb.overview()``        — what's in the space (an agent calls this to orient).
* ``kb.search(...)``       — semantic search returning flat, cited hits.
* ``kb.get_document(...)`` — one document's full text + metadata.
* ``kb.openai_schema()``   — raw tool specs for the bare Chat Completions / Messages API.
* ``kb.call(name, args)``  — the single dispatch path every framework adapter funnels through.

To plug into a real framework, install its extra and swap the last block for one line::

    pip install "indx[openai-agents]"   # then: kb.openai()
    pip install "indx[langchain]"       # then: kb.langchain() / kb.langchain_retriever()
    pip install "indx[pydantic-ai]"     # then: kb.pydantic_ai()
    pip install "indx[claude-agent]"    # then: kb.claude()
    pip install "indx[mcp]"             # then: indx mcp <archive>  (Mastra, Cursor, Claude Desktop)

Run it with the project interpreter::

    .venv/bin/python examples/agent_quickstart.py
"""

from __future__ import annotations

import json
from pathlib import Path

from indx.agent import connect
from indx.pipeline import DirectoryPipeline

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_space"


def main() -> None:
    # Build a knowledge space in memory with the offline core stack (no extras, no API key).
    space = DirectoryPipeline(
        parser="plaintext", embedder="hash", store="jsonl", enrich=False, seed=0
    ).run(SAMPLE_DIR, name="quickstart")

    # Plug it in: one call loads the "USB drive" into an agent-ready connector.
    kb = connect(space, name="quickstart")

    # 1) Orient — what is this space about?
    overview = kb.overview(sample=5)
    print(f"overview: {overview.documents} docs, {overview.chunks} chunks, types={overview.types}")
    for card in overview.sample_documents:
        print(f"  - {card.path} [{card.doc_type or 'unknown'}]")

    # 2) Search — grounded, cited hits.
    print("\nsearch: 'How does indx stay deterministic and offline?'")
    results = kb.search("How does indx stay deterministic and offline?", k=3)
    for rank, hit in enumerate(results.hits, start=1):
        print(f"  {rank}. score={hit.score:.3f}  {hit.source}  {hit.text[:60]!r}")

    # 3) Read a source in full.
    top = results.hits[0]
    detail = kb.get_document(top.source or top.document_id)
    if detail is not None:
        print(f"\nget_document({detail.path!r}): {detail.chunk_count} chunk(s), "
              f"{len(detail.text)} chars")

    # 4) Raw tool specs for a framework-less wiring (Chat Completions / Messages API).
    print("\nopenai_schema tool names:", [t["function"]["name"] for t in kb.openai_schema()])

    # 5) The dispatch path every framework adapter uses under the hood.
    payload = kb.call("indx_search", {"query": "offline core", "k": 1})
    print("call('indx_search'):", json.dumps(payload)[:80], "…")


if __name__ == "__main__":
    main()
