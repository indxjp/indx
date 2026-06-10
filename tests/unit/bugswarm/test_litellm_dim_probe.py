"""Regression: LiteLLMEmbedder.dim must reflect the provider's true width *before* the first
real embed, so the pipeline pre-sizes dimension-locked stores (qdrant/pgvector/…) correctly.

Previously ``dim`` stayed at the 1536 placeholder until the first ``embed()`` call, so a non-1536
provider (ollama=768, cohere=1024, titan-v2=1024/512/256) created a wrong-width store that later
rejected the real vectors. ``dim`` is now resolved on first read via a single-token probe, so the
store-sizing path and the manifest both see the real width even for an empty corpus.

The vendor ``litellm`` SDK is mocked so this runs fully offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import indx.embed.litellm  # noqa: F401  -- force-load before any require_extra patch
from indx.embed.litellm import LiteLLMEmbedder


class _Response:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


@pytest.fixture
def fake_litellm_768(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A fake provider that returns 768-dim vectors (e.g. ollama/nomic-embed-text)."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr("indx.embed.litellm.require_extra", lambda *a, **k: None)
    module = types.ModuleType("litellm")
    module.calls = []  # type: ignore[attr-defined]

    def embedding(**kwargs: Any) -> _Response:
        module.calls.append(kwargs)  # type: ignore[attr-defined]
        return _Response(
            [{"embedding": [0.0] * 768, "index": i} for i in range(len(kwargs["input"]))]
        )

    module.embedding = embedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def test_dim_resolved_before_first_real_embed(fake_litellm_768: types.ModuleType) -> None:
    # The pipeline reads embedder.dim to pre-size a store *before* embedding any document.
    emb = LiteLLMEmbedder(model="ollama/nomic-embed-text")
    assert emb.dim == 768  # probed on read, not the stale 1536 placeholder
    # The probe used a single token, not the (not-yet-known) corpus.
    assert fake_litellm_768.calls[0]["input"] == ["probe"]


def test_pinned_dim_skips_probe(fake_litellm_768: types.ModuleType) -> None:
    emb = LiteLLMEmbedder(model="ollama/nomic-embed-text", dim=512)
    assert emb.dim == 512
    assert fake_litellm_768.calls == []  # no probe issued when pinned


def test_empty_corpus_dim_is_real_width_not_placeholder(
    fake_litellm_768: types.ModuleType,
) -> None:
    # Empty corpus: embed() is never called by the pipeline, yet the manifest still stamps
    # embedder.dim. Reading dim must probe and report the real width, never the 1536 placeholder.
    emb = LiteLLMEmbedder(model="ollama/nomic-embed-text")
    assert emb.embed([]) == []
    assert emb.dim == 768


def test_real_embed_after_probe_keeps_width(fake_litellm_768: types.ModuleType) -> None:
    emb = LiteLLMEmbedder(model="ollama/nomic-embed-text")
    assert emb.dim == 768
    vectors = emb.embed(["a", "b"])
    assert [len(v) for v in vectors] == [768, 768]
    assert emb.dim == 768
