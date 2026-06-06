"""Harness for the **live** tier (testing-strategy §3.5, the ``live`` marker).

These tests hit *real* third-party services (OpenAI, Anthropic, Cohere, LlamaParse,
Azure OpenAI) and therefore need real API keys. They are excluded from the default
offline run (``addopts = -m 'not ... and not live'``) and only execute when:

* the ``live`` marker is explicitly selected (``pytest -m live``), and
* the credential(s) a given service needs are present in the environment.

Each test self-skips when its key is absent, so a single ``pytest -m live`` run
exercises exactly the services whose secrets the CI job was able to inject — which is
the behaviour the e2e workflow relies on (run the keys that are in Secrets, skip the
rest). Inputs are deliberately tiny (a two-file tree, one-word prompts) to keep the
real-API cost and wall-clock of a CI run negligible.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from indx.core.chunk import Chunk
from indx.store.base import VectorStore

# A text-bearing raster used by the cloud OCR parser e2e (Textract/Document Intelligence/
# Document AI) and as a real (non-1x1) image for the cloud VLM e2e. Rendered for indx; see
# tests/corpus/PROVENANCE.md. Textract's synchronous DetectDocumentText only accepts PNG/JPEG
# (PDF needs the async S3 flow), so a raster — not a PDF — is the portable choice here.
OCR_SAMPLE = Path(__file__).resolve().parents[1] / "corpus" / "ocr" / "sample.png"


@pytest.fixture
def require_env() -> Callable[..., dict[str, str]]:
    """Return a helper that skips the test unless every named env var is non-empty.

    Exposed as a fixture (rather than a bare import) so test modules don't depend on the
    conftest being importable as a package. Usage::

        def test_x(require_env):
            require_env("OPENAI_API_KEY")

    The ``pytest.skip`` message names the missing variable(s) so a half-configured Secrets
    store produces an actionable skip line instead of a confusing vendor auth error. The
    helper returns the resolved values so a test can use them directly.
    """

    def _require(*names: str) -> dict[str, str]:
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            pytest.skip(f"live service credentials absent: {', '.join(missing)}")
        return {n: os.environ[n] for n in names}

    return _require


@pytest.fixture
def live_corpus(tmp_path: Path) -> Path:
    """A tiny two-file directory tree fed to the real pipeline.

    Small on purpose: just enough text for an embedder/parser to produce real vectors or
    blocks and for retrieval to rank one file above the other, without spending tokens.
    """
    handbook = tmp_path / "handbook"
    handbook.mkdir()
    (handbook / "onboarding.md").write_text(
        "# Onboarding\n\nNew hires complete payroll setup and choose benefits in week one.\n"
    )
    engineering = tmp_path / "engineering"
    engineering.mkdir()
    (engineering / "search-api.md").write_text(
        "# Search API\n\nThe service exposes a REST endpoint that ranks documents by relevance.\n"
    )
    return tmp_path


@pytest.fixture
def image_bytes() -> bytes:
    """Raw bytes of a real (non-1x1) PNG fed to cloud VLM ``describe`` calls.

    A 1x1 pixel is enough for some vision APIs but is rejected by others that enforce a
    minimum image size, so the cloud VLM e2e uses this genuine raster instead.
    """
    return OCR_SAMPLE.read_bytes()


@pytest.fixture
def ocr_doc(tmp_path: Path) -> Path:
    """A single-file tree holding the OCR sample image, for cloud parser builds.

    The image renders the literal phrase "Quarterly revenue grew across every region.", so a
    test can assert the OCR recovered a known token (e.g. ``"revenue"``) end to end.
    """
    dest = tmp_path / "scan.png"
    dest.write_bytes(OCR_SAMPLE.read_bytes())
    return tmp_path


@pytest.fixture
def cloud_store_roundtrip() -> Callable[[VectorStore], None]:
    """Return a helper that exercises a cloud store's upsert → search → delete contract.

    Same conformance bar as the embedded-store extras test, but tolerant of the
    *eventual consistency* every hosted vector store exhibits: a just-upserted document is
    not always queryable on the very next call (Azure AI Search / OpenSearch Serverless /
    BigQuery streaming buffer / Vertex index refresh all lag seconds). So ``search`` is
    retried with backoff until the nearest neighbour appears, rather than asserted once.

    The orthogonal 2-D vectors make the nearest neighbour unambiguous (cosine), so the
    assertion is deterministic once the write is visible. ``delete`` is exercised but its
    propagation is *not* hard-asserted: a hosted delete can lag past any sane test budget,
    and a nightly run should not flake on deletion latency.
    """

    def _chunk(cid: str, vec: list[float]) -> Chunk:
        return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)

    def _search_until(store: VectorStore, vec: list[float], k: int, want: str) -> list:
        # Up to ~30s of backoff: enough for index refresh without stalling the suite.
        delay = 1.0
        hits: list = []
        for _ in range(8):
            hits = store.search(vec, k=k)
            if want in [h.chunk.id for h in hits]:
                return hits
            time.sleep(delay)
            delay = min(delay * 1.6, 6.0)
        return hits

    def _exercise(store: VectorStore) -> None:
        store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

        hits = _search_until(store, [1.0, 0.0], k=2, want="a")
        ids = [h.chunk.id for h in hits]
        assert "a" in ids, f"upserted nearest neighbour never became queryable; got {ids}"
        assert ids[0] == "a", f"nearest neighbour should rank first; got {ids}"
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)), (
            "hits must be in descending score order"
        )

        # Exercise delete for coverage; do not hard-assert propagation (hosted lag).
        store.delete(["a"])

    return _exercise
