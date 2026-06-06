"""Live e2e against the real AWS cloud stack (``indx[aws]``).

Exercises every AWS adapter against real services, each test self-skipping unless the
credentials/resources it needs are present (so a partially-provisioned account still gets a
green run for the pieces it has):

* **bedrock** LLM / VLM / embedder — Amazon Bedrock Runtime (Converse + ``invoke_model``).
  Needs AWS credentials, a region, and **model access enabled** for the default Claude /
  Titan model ids in that region (enable under Bedrock → Model access).
* **textract** parser — Amazon Textract synchronous ``DetectDocumentText`` (PNG/JPEG).
* **s3vectors** store — Amazon S3 Vectors; needs a pre-created vector bucket
  (``AWS_S3_VECTORS_BUCKET``) and ``boto3 >= 1.40``.

Credentials ride boto3's own chain (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
profile / role); the region comes from ``AWS_REGION``. Inputs are deliberately tiny to keep
the real-API cost negligible.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder, get_llm, get_store, get_vlm
from indx.store.base import VectorStore

pytestmark = pytest.mark.live

# Every Bedrock leg needs the boto3 chain plus a region; share one gate.
_BEDROCK_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")


def test_bedrock_llm_completion(require_env) -> None:
    require_env(*_BEDROCK_ENV)

    llm = get_llm("bedrock")  # default: a us.-prefixed Claude 3.5 Sonnet inference profile
    reply = llm.complete("Reply with exactly one word: pong", max_tokens=16)

    assert isinstance(reply, str)
    assert reply.strip(), "live Bedrock completion returned empty text"


def test_bedrock_vlm_describe(require_env, image_bytes: bytes) -> None:
    require_env(*_BEDROCK_ENV)

    vlm = get_vlm("bedrock")
    caption = vlm.describe(image_bytes, prompt="Reply with one word describing this image.")

    assert isinstance(caption, str)
    assert caption.strip(), "live Bedrock vision describe returned empty text"


def test_bedrock_embedder_direct(require_env) -> None:
    require_env(*_BEDROCK_ENV)

    embedder = get_embedder("bedrock")  # default: amazon.titan-embed-text-v2:0 (1024-dim)
    vectors = embedder.embed(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    width = len(vectors[0])
    assert width > 0 and len(vectors[1]) == width, "ragged/empty embeddings"
    assert all(isinstance(x, float) for x in vectors[0])


def test_bedrock_embedder_build_and_query(require_env, live_corpus: Path) -> None:
    require_env(*_BEDROCK_ENV)

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="bedrock", store="jsonl"
    ).run(live_corpus)

    assert space.chunks, "build produced no chunks"
    vectors = [c.embedding for c in space.chunks if c.embedding is not None]
    assert len(vectors) == len(space.chunks), "every chunk should carry a real embedding"
    width = len(vectors[0])
    assert width > 0 and all(len(v) == width for v in vectors), "ragged/empty embeddings"

    hits = space.search("how do new hires set up payroll", k=2)
    assert hits, "live retrieval returned no hits"
    assert hits[0].source is not None and hits[0].source.path


def test_textract_parser_build(require_env, ocr_doc: Path) -> None:
    require_env(*_BEDROCK_ENV)

    space = DirectoryPipeline(
        seed=0, parser="textract", llm="none", embedder="hash", store="jsonl"
    ).run(ocr_doc)

    assert space.chunks, "Textract build produced no chunks"
    text = " ".join(c.text for c in space.chunks).lower()
    # The fixture image renders "Quarterly revenue grew across every region."
    assert "revenue" in text, f"OCR did not recover expected text; got: {text!r}"


def test_s3vectors_store_round_trip(
    require_env, cloud_store_roundtrip: Callable[[VectorStore], None]
) -> None:
    env = require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
    bucket = require_env("AWS_S3_VECTORS_BUCKET")["AWS_S3_VECTORS_BUCKET"]

    store = get_store("s3vectors", bucket=bucket, region=env["AWS_REGION"], dim=2)
    cloud_store_roundtrip(store)
