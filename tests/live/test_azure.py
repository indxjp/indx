"""Live e2e against the real Azure cloud stack (``indx[azure]``).

Azure is a single extra spanning several backends, each keyed on its own service's
variables and self-skipping when they are absent:

* **azure** LLM / VLM / embedder — Azure OpenAI deployments. Shared resource
  (``AZURE_OPENAI_API_KEY`` / ``AZURE_OPENAI_ENDPOINT``) plus a *per-modality deployment name*
  (``AZURE_OPENAI_DEPLOYMENT`` for chat, ``AZURE_OPENAI_VLM_DEPLOYMENT`` for vision,
  ``AZURE_OPENAI_EMBED_DEPLOYMENT`` for embeddings).
* **docintel** parser — Azure AI Document Intelligence
  (``AZURE_DOCUMENTINTELLIGENCE_ENDPOINT`` / ``AZURE_DOCUMENTINTELLIGENCE_KEY``).
* **azure-search** store — Azure AI Search (``AZURE_SEARCH_SERVICE_ENDPOINT`` /
  ``AZURE_SEARCH_API_KEY`` / ``AZURE_SEARCH_INDEX_NAME``).

The scaffold Enrich stage is LLM/VLM-free, so the chat/vision adapters are driven directly
through the public registry — the same objects the pipeline would use.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder, get_llm, get_store, get_vlm
from indx.store.base import VectorStore

pytestmark = pytest.mark.live

# Shared Azure OpenAI resource (key + endpoint); each modality adds its own deployment name.
_AOAI_ENV = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")


def test_azure_llm_completion(require_env) -> None:
    require_env(*_AOAI_ENV, "AZURE_OPENAI_DEPLOYMENT")

    # The adapter reads the deployment/endpoint/key/version from the environment.
    llm = get_llm("azure")
    reply = llm.complete("Reply with exactly one word: pong", max_tokens=512)

    assert isinstance(reply, str)
    assert reply.strip(), "live Azure OpenAI completion returned empty text"


def test_azure_vlm_describe(require_env, image_bytes: bytes) -> None:
    require_env(*_AOAI_ENV, "AZURE_OPENAI_VLM_DEPLOYMENT")

    vlm = get_vlm("azure")
    caption = vlm.describe(image_bytes, prompt="Reply with one word describing this image.")

    assert isinstance(caption, str)
    assert caption.strip(), "live Azure OpenAI vision describe returned empty text"


def test_azure_embedder_direct(require_env) -> None:
    require_env(*_AOAI_ENV, "AZURE_OPENAI_EMBED_DEPLOYMENT")

    embedder = get_embedder("azure")
    vectors = embedder.embed(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    width = len(vectors[0])
    assert width > 0 and len(vectors[1]) == width, "ragged/empty embeddings"
    assert all(isinstance(x, float) for x in vectors[0])


def test_azure_embedder_build_and_query(require_env, live_corpus: Path) -> None:
    require_env(*_AOAI_ENV, "AZURE_OPENAI_EMBED_DEPLOYMENT")

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="azure", store="jsonl"
    ).run(live_corpus)

    assert space.chunks, "build produced no chunks"
    vectors = [c.embedding for c in space.chunks if c.embedding is not None]
    assert len(vectors) == len(space.chunks), "every chunk should carry a real embedding"
    width = len(vectors[0])
    assert width > 0 and all(len(v) == width for v in vectors), "ragged/empty embeddings"

    hits = space.search("how do new hires set up payroll", k=2)
    assert hits, "live retrieval returned no hits"
    assert hits[0].source is not None and hits[0].source.path


def test_docintel_parser_build(require_env, ocr_doc: Path) -> None:
    require_env("AZURE_DOCUMENTINTELLIGENCE_ENDPOINT", "AZURE_DOCUMENTINTELLIGENCE_KEY")

    space = DirectoryPipeline(
        seed=0, parser="docintel", llm="none", embedder="hash", store="jsonl"
    ).run(ocr_doc)

    assert space.chunks, "Document Intelligence build produced no chunks"
    text = " ".join(c.text for c in space.chunks).lower()
    # The fixture image renders "Quarterly revenue grew across every region."
    assert "revenue" in text, f"OCR did not recover expected text; got: {text!r}"


def test_azure_search_store_round_trip(
    require_env, cloud_store_roundtrip: Callable[[VectorStore], None]
) -> None:
    require_env("AZURE_SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME")

    store = get_store("azure-search", dim=2)
    cloud_store_roundtrip(store)
