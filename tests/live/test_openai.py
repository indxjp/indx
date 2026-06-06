"""Live e2e against the real OpenAI API (``OPENAI_API_KEY``).

Covers the two places OpenAI plugs into indx:

* **Embedder** — exercised by a *full pipeline build* (``--embedder openai``). The
  embed-pack stage calls ``embeddings.create`` for real, so this is a genuine
  directory-in → vectors-out e2e, and ``space.search`` re-embeds the query against the
  same live model.
* **LLM** — the default scaffold Enrich stage is LLM-free, so a build never calls the
  chat API. We exercise the ``openai`` LLM adapter directly through the public registry,
  which is the same object the pipeline would use, proving the chat round-trip works.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from indx import DirectoryPipeline
from indx.registry import get_embedder, get_llm, get_vlm

pytestmark = pytest.mark.live

_OPENAI_EMBED_DIM = 1536  # text-embedding-3-small


def _solid_png(size: int = 64, rgb: tuple[int, int, int] = (30, 90, 200)) -> bytes:
    """Build a valid solid-color RGB PNG of ``size``×``size`` with no third-party deps.

    A literal 1×1 PNG is rejected by the OpenAI vision API ("unsupported image"), so the VLM
    test needs a real, decodable image. Generating one from stdlib (``zlib``/``struct``) keeps
    a binary asset out of the repo while still producing bytes the API accepts.
    """
    raw = (bytes([0]) + bytes(rgb) * size) * size  # one filter byte (0) per scanline

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit, color-type 2 (RGB)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


_TINY_PNG = _solid_png()


def test_openai_embedder_build_and_query(require_env, live_corpus: Path) -> None:
    require_env("OPENAI_API_KEY")

    space = DirectoryPipeline(
        seed=0, parser="plaintext", llm="none", embedder="openai", store="jsonl"
    ).run(live_corpus)

    assert space.chunks, "build produced no chunks"
    vectors = [c.embedding for c in space.chunks if c.embedding is not None]
    assert len(vectors) == len(space.chunks), "every chunk should carry a real embedding"
    assert all(len(v) == _OPENAI_EMBED_DIM for v in vectors), "unexpected embedding width"

    hits = space.search("how do new hires set up payroll", k=2)
    assert hits, "live retrieval returned no hits"
    # Provenance must survive into the SearchHit (hit.source is a shorthand for chunk.source).
    assert hits[0].source is not None
    assert hits[0].source.path


def test_openai_llm_completion(require_env) -> None:
    require_env("OPENAI_API_KEY")

    llm = get_llm("openai")  # default model: gpt-5-mini
    reply = llm.complete("Reply with exactly one word: pong", max_tokens=512)

    assert isinstance(reply, str)
    assert reply.strip(), "live OpenAI completion returned empty text"


def test_openai_embedder_direct(require_env) -> None:
    require_env("OPENAI_API_KEY")

    embedder = get_embedder("openai")
    vectors = embedder.embed(["alpha beta", "gamma delta"])

    assert len(vectors) == 2
    assert all(len(v) == _OPENAI_EMBED_DIM for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_gpt4o_vlm_describe(require_env) -> None:
    require_env("OPENAI_API_KEY")

    # The Enrich stage is VLM-free (records the VLM for provenance only), so the gpt4o
    # vision backend is exercised directly through the registry — the same object a custom
    # VLM-enrich stage would use.
    vlm = get_vlm("gpt4o")
    caption = vlm.describe(_TINY_PNG, prompt="Reply with one word describing this image.")

    assert isinstance(caption, str)
    assert caption.strip(), "live gpt4o vision describe returned empty text"
