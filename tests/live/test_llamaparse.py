"""Live e2e against the real LlamaParse / LlamaCloud API (``LLAMA_CLOUD_API_KEY``).

LlamaParse is a parser backend, so a *full pipeline build* (``--parser llamaparse``)
uploads and parses the source file for real. The rest of the stack is pinned to the
offline core (``hash`` embedder, ``jsonl`` store) so this test isolates the parser and
needs only the LlamaCloud key — no OpenAI key required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline

pytestmark = pytest.mark.live


@pytest.fixture
def single_doc(tmp_path: Path) -> Path:
    """One small markdown file — minimises LlamaParse upload/credit cost."""
    (tmp_path / "note.md").write_text(
        "# Quarterly Note\n\n"
        "Revenue grew across every region. "
        "The engineering team shipped the new search ranking model.\n"
    )
    return tmp_path


def test_llamaparse_build(require_env, single_doc: Path) -> None:
    require_env("LLAMA_CLOUD_API_KEY")

    space = DirectoryPipeline(
        seed=0, parser="llamaparse", llm="none", embedder="hash", store="jsonl"
    ).run(single_doc)

    assert space.documents(), "LlamaParse build produced no documents"
    assert space.chunks, "LlamaParse build produced no chunks"
    # The parser must have recovered real text from the uploaded file.
    assert any(c.text.strip() for c in space.chunks), "parsed chunks are all empty"

    hits = space.search("search ranking model", k=2)
    assert hits and hits[0].source is not None and hits[0].source.path
