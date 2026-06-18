"""Unit tests for VertexEmbedder — vendor SDK mocked, fully offline (coding-standards §11)."""

from __future__ import annotations

import math
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.vertex import VertexEmbedder
from indx.errors import MissingExtraError, StageError

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeEmbedContentConfig:
    """Stand-in for ``types.EmbedContentConfig`` recording its config kwargs."""

    def __init__(self, *, task_type: str, output_dimensionality: int) -> None:
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality


def _fake_vector(text: str, dim: int) -> list[float]:
    """Deterministic, *non-unit* vector: encodes ``len(text)`` in the leading
    component (1.0 elsewhere) so input order stays recoverable after the adapter's
    L2 normalization, while keeping the pre-normalization norm != 1 so the
    normalization step is observable."""
    return [float(len(text))] + [1.0] * (dim - 1)


class _FakeModels:
    """Returns a deterministic vector per input; preserves input order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embed_content(
        self, *, model: str, contents: Sequence[str], config: _FakeEmbedContentConfig
    ) -> _FakeResponse:
        self.calls.append(
            {
                "model": model,
                "contents": list(contents),
                "task_type": config.task_type,
                "output_dimensionality": config.output_dimensionality,
            }
        )
        embeddings = [
            _FakeEmbedding(_fake_vector(text, config.output_dimensionality)) for text in contents
        ]
        return _FakeResponse(embeddings)


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.models = _FakeModels()


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``google.genai`` namespace and neutralize the dependency gate."""
    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    # The adapter binds require_extra by name (``from ... import``), so neutralize the
    # binding it actually calls too.
    monkeypatch.setattr("indx.embed.vertex.require_extra", lambda *a, **k: None)

    # Namespace package: register every parent in sys.modules.
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    genai_mod.Client = _FakeClient  # type: ignore[attr-defined]
    types_mod.EmbedContentConfig = _FakeEmbedContentConfig  # type: ignore[attr-defined]
    google_mod.genai = genai_mod  # type: ignore[attr-defined]
    genai_mod.types = types_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return genai_mod


def test_satisfies_embedder_protocol(fake_genai: types.ModuleType) -> None:
    from indx.embed.base import Embedder

    assert isinstance(VertexEmbedder(), Embedder)


def test_embed_returns_one_vector_per_text_in_order(fake_genai: types.ModuleType) -> None:
    embedder = VertexEmbedder()
    texts = ["a", "bb", "ccc"]
    vectors = embedder.embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == VertexEmbedder.dim for v in vectors)
    # Fake encodes len(text) into the leading component, so after normalization the
    # leading component is strictly increasing in len(text): input order is verifiable.
    leading = [v[0] for v in vectors]
    assert leading == sorted(leading) and len(set(leading)) == len(leading)


def test_embed_empty_returns_empty_without_calling_api(fake_genai: types.ModuleType) -> None:
    embedder = VertexEmbedder()
    assert embedder.embed([]) == []
    assert embedder._client.models.calls == []  # type: ignore[attr-defined]


def test_default_dim_is_768(fake_genai: types.ModuleType) -> None:
    embedder = VertexEmbedder()
    assert embedder.dim == 768
    vectors = embedder.embed(["a"])
    assert len(vectors[0]) == 768
    assert embedder._client.models.calls[0]["output_dimensionality"] == 768  # type: ignore[attr-defined]


def test_dim_configurable(fake_genai: types.ModuleType) -> None:
    for d in (1536, 3072):
        embedder = VertexEmbedder(dim=d)
        assert embedder.dim == d
        assert len(embedder.embed(["x"])[0]) == d


def test_task_type_default_and_override(fake_genai: types.ModuleType) -> None:
    embedder = VertexEmbedder()
    embedder.embed(["a"])
    assert embedder._client.models.calls[0]["task_type"] == "RETRIEVAL_DOCUMENT"  # type: ignore[attr-defined]

    query = VertexEmbedder(task_type="RETRIEVAL_QUERY")
    query.embed(["a"])
    assert query._client.models.calls[0]["task_type"] == "RETRIEVAL_QUERY"  # type: ignore[attr-defined]


def test_gemini_embedding_sends_one_text_per_request(fake_genai: types.ModuleType) -> None:
    # gemini-embedding-* accepts only ONE input text per request on Vertex (a multi-text
    # request 400s), so the adapter must clamp to 1 regardless of batch_size — even when a
    # larger batch_size is configured. Order and one-vector-per-input must still hold.
    embedder = VertexEmbedder(batch_size=8)
    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert len(vectors) == 5
    leading = [v[0] for v in vectors]
    assert leading == sorted(leading) and len(set(leading)) == len(leading)
    calls = embedder._client.models.calls  # type: ignore[attr-defined]
    assert [len(c["contents"]) for c in calls] == [1, 1, 1, 1, 1]


def test_non_gemini_model_respects_batch_size(fake_genai: types.ModuleType) -> None:
    # The single-text clamp is specific to the gemini-embedding-* family; the older
    # text-embedding-* models batch up to ~250, so a configured batch_size is honored.
    embedder = VertexEmbedder(model="text-embedding-005", batch_size=2)
    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert len(vectors) == 5
    leading = [v[0] for v in vectors]
    assert leading == sorted(leading) and len(set(leading)) == len(leading)
    calls = embedder._client.models.calls  # type: ignore[attr-defined]
    assert [len(c["contents"]) for c in calls] == [2, 2, 1]


def test_name_records_actual_model(fake_genai: types.ModuleType) -> None:
    assert VertexEmbedder().name == "gemini-embedding-001"
    assert VertexEmbedder(model="custom-embed-002").name == "custom-embed-002"


def test_embed_returns_l2_normalized_vectors(fake_genai: types.ModuleType) -> None:
    # gemini-embedding-001 only guarantees unit-norm at its native 3072 width; the
    # default 768 (Matryoshka-truncated) width is non-unit and must be re-normalized at
    # the adapter edge for cross-backend cosine consistency (matches e5 / HashEmbedder).
    embedder = VertexEmbedder()
    for vec in embedder.embed(["a", "bb", "ccc"]):
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("bad_dim", [512, 999, 0, 1024])
def test_unsupported_dim_raises_stage_error(fake_genai: types.ModuleType, bad_dim: int) -> None:
    # Only 768/1536/3072 are valid output_dimensionality values; an unsupported width
    # must fail fast at construction rather than desyncing the manifest dim (mirrors
    # Bedrock's Titan-v2 dim guard).
    with pytest.raises(StageError):
        VertexEmbedder(dim=bad_dim)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization here: the real (dep-absent) environment must raise.
    with pytest.raises(MissingExtraError):
        VertexEmbedder()
