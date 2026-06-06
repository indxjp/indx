"""Live e2e against the real GCP cloud stack (``indx[gcp]``).

Exercises every GCP adapter against real services, each test self-skipping unless the
credentials/resources it needs are present:

* **vertex** LLM / VLM / embedder — Vertex AI Gemini via the unified ``google-genai`` SDK.
* **docai** parser — Google Document AI; needs a pre-created processor
  (``DOCAI_PROCESSOR_ID``) in the Document AI multi-region (``DOCAI_LOCATION``, ``us``/``eu``).
* **bigquery** store — BigQuery vector search; needs an existing dataset (``BIGQUERY_DATASET``)
  in the project.

Auth is Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS`` / attached SA);
project and region come from ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``. No API key
is ever read here. Inputs are deliberately tiny to keep the real-API cost negligible.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder, get_llm, get_store, get_vlm
from indx.store.base import VectorStore

pytestmark = pytest.mark.live

# Every Vertex/GCP leg needs a project plus ADC; share one gate.
_GCP_ENV = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS")


def test_vertex_llm_completion(require_env) -> None:
    require_env(*_GCP_ENV)

    llm = get_llm("vertex")  # default: gemini-2.5-flash
    # Gemini 2.5 is a *thinking* model: a tiny max_output_tokens budget is spent on
    # reasoning, leaving no visible text. Give it room so the one-word answer survives.
    reply = llm.complete("Reply with exactly one word: pong", max_tokens=256)

    assert isinstance(reply, str)
    assert reply.strip(), "live Vertex completion returned empty text"


def test_vertex_vlm_describe(require_env, image_bytes: bytes) -> None:
    require_env(*_GCP_ENV)

    vlm = get_vlm("vertex")
    caption = vlm.describe(image_bytes, prompt="Reply with one word describing this image.")

    assert isinstance(caption, str)
    assert caption.strip(), "live Vertex vision describe returned empty text"


def test_vertex_embedder_direct(require_env) -> None:
    require_env(*_GCP_ENV)

    embedder = get_embedder("vertex")  # default: gemini-embedding-001 truncated to 768
    vectors = embedder.embed(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    assert all(len(v) == embedder.dim for v in vectors), "unexpected embedding width"
    assert all(isinstance(x, float) for x in vectors[0])


def test_vertex_embedder_build_and_query(require_env, live_corpus: Path) -> None:
    require_env(*_GCP_ENV)

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="vertex", store="jsonl"
    ).run(live_corpus)

    assert space.chunks, "build produced no chunks"
    vectors = [c.embedding for c in space.chunks if c.embedding is not None]
    assert len(vectors) == len(space.chunks), "every chunk should carry a real embedding"
    width = len(vectors[0])
    assert width > 0 and all(len(v) == width for v in vectors), "ragged/empty embeddings"

    hits = space.search("how do new hires set up payroll", k=2)
    assert hits, "live retrieval returned no hits"
    assert hits[0].source is not None and hits[0].source.path


def test_docai_parser_build(require_env, ocr_doc: Path) -> None:
    require_env(*_GCP_ENV)
    processor_id = require_env("DOCAI_PROCESSOR_ID")["DOCAI_PROCESSOR_ID"]
    from indx.parsers.docai import DocumentAIParser

    # The Document AI multi-region ("us"/"eu") is distinct from a compute region; pass it
    # explicitly. project/credentials come from the environment / ADC.
    parser = DocumentAIParser(
        location=os.environ.get("DOCAI_LOCATION", "us"),
        processor_id=processor_id,
    )
    space = DirectoryPipeline(
        seed=0, parser=parser, llm="none", embedder="hash", store="jsonl"
    ).run(ocr_doc)

    assert space.chunks, "Document AI build produced no chunks"
    text = " ".join(c.text for c in space.chunks).lower()
    # The fixture image renders "Quarterly revenue grew across every region."
    assert "revenue" in text, f"OCR did not recover expected text; got: {text!r}"


def test_bigquery_store_round_trip(
    require_env, cloud_store_roundtrip: Callable[[VectorStore], None]
) -> None:
    project = require_env(*_GCP_ENV)["GOOGLE_CLOUD_PROJECT"]
    dataset = require_env("BIGQUERY_DATASET")["BIGQUERY_DATASET"]

    store = get_store("bigquery", project=project, dataset=dataset, dim=2)
    cloud_store_roundtrip(store)
