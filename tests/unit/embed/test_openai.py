"""Unit tests for OpenAIEmbedder — vendor SDK mocked, fully offline (coding-standards §11)."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.openai import OpenAIEmbedder
from indx.errors import MissingExtraError

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeEmbedding:
    def __init__(self, embedding: list[float], index: int) -> None:
        self.embedding = embedding
        self.index = index


class _FakeResponse:
    def __init__(self, data: list[_FakeEmbedding]) -> None:
        self.data = data


class _FakeEmbeddings:
    """Returns a deterministic vector per input; preserves input order via ``index``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, *, input: Sequence[str], model: str) -> _FakeResponse:
        self.calls.append({"input": list(input), "model": model})
        data = [
            _FakeEmbedding([float(len(text))] * OpenAIEmbedder.dim, index=i)
            for i, text in enumerate(input)
        ]
        return _FakeResponse(data)


class _FakeOpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.embeddings = _FakeEmbeddings()


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``openai`` module and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter binds require_extra by name (``from ... import``), so neutralize the
    # binding it actually calls too.
    monkeypatch.setattr("indx.embed.openai.require_extra", lambda *a, **k: None)
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def test_satisfies_embedder_protocol(fake_openai: types.ModuleType) -> None:
    from indx.embed.base import Embedder

    assert isinstance(OpenAIEmbedder(), Embedder)


def test_embed_returns_one_vector_per_text_in_order(fake_openai: types.ModuleType) -> None:
    embedder = OpenAIEmbedder()
    texts = ["a", "bb", "ccc"]
    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == OpenAIEmbedder.dim for v in vectors)
    # Fake encodes len(text) into each component, so input order is verifiable.
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_embed_empty_returns_empty_without_calling_api(fake_openai: types.ModuleType) -> None:
    embedder = OpenAIEmbedder()
    assert embedder.embed([]) == []
    assert embedder._client.embeddings.calls == []  # type: ignore[attr-defined]


def test_batching_splits_into_multiple_requests(fake_openai: types.ModuleType) -> None:
    embedder = OpenAIEmbedder(batch_size=2)
    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    calls = embedder._client.embeddings.calls  # type: ignore[attr-defined]
    assert [len(c["input"]) for c in calls] == [2, 2, 1]


def test_token_budget_splits_dense_text_under_item_cap(fake_openai: types.ModuleType) -> None:
    # Regression: dense (e.g. CJK) text overflowed OpenAI's 300k-token/request cap because
    # batching was by item count only. With a tiny token budget, a high item cap still splits.
    embedder = OpenAIEmbedder(batch_size=256, max_tokens_per_request=30)
    texts = ["日本語のテスト文章" * 2, "もう一つの長い文章" * 2, "三つ目"]
    vectors = embedder.embed(texts)

    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]  # order preserved
    calls = embedder._client.embeddings.calls  # type: ignore[attr-defined]
    assert len(calls) > 1  # split across multiple requests instead of one oversized one


def test_name_records_actual_model(fake_openai: types.ModuleType) -> None:
    assert OpenAIEmbedder().name == "text-embedding-3-small"
    assert OpenAIEmbedder(model="text-embedding-3-large").name == "text-embedding-3-large"


def test_dim_reflects_selected_model(fake_openai: types.ModuleType) -> None:
    # text-embedding-3-large is 3072-dim; the default (3-small) stays 1536-dim.
    assert OpenAIEmbedder(model="text-embedding-3-large").dim == 3072
    assert OpenAIEmbedder().dim == 1536


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        OpenAIEmbedder()
