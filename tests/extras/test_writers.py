"""Extras e2e for the optional output WRITER backends, against their real libraries.

Each writer converts a built space into its ecosystem's document type and serializes one
object per chunk to ``<out>/<name>.jsonl``. Exercised by a full offline-core build
(``hash`` embedder — no key, no network) whose ``run(out=...)`` seals the space through the
chosen writer; we then assert the file exists and its rows carry the ecosystem's payload
field.

* ``langchain``  — ``langchain-core`` ``Document`` (``page_content``).
* ``llamaindex`` — ``llama-index-core`` ``TextNode`` (``text``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indx import DirectoryPipeline

pytestmark = pytest.mark.extras


def _build_to(writer: str, corpus: Path, out: Path) -> list[dict]:
    DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl", output=writer
    ).run(corpus, out, name="handbook")

    sidecar = out / "handbook.jsonl"
    assert sidecar.exists(), f"{writer} writer produced no {sidecar.name}"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    assert rows, f"{writer} writer wrote an empty file"
    return rows


def test_langchain_writer(requires_lib, text_corpus: Path, tmp_path: Path) -> None:
    requires_lib("langchain_core")
    rows = _build_to("langchain", text_corpus, tmp_path / "out")
    assert all("page_content" in r for r in rows), "every LangChain Document needs page_content"


def test_llamaindex_writer(requires_lib, text_corpus: Path, tmp_path: Path) -> None:
    requires_lib("llama_index")
    rows = _build_to("llamaindex", text_corpus, tmp_path / "out")
    assert all("text" in r for r in rows), "every LlamaIndex TextNode needs a text field"
