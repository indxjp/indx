"""Mocked unit tests for :class:`~indx.embed.bge_m3.BGEM3Embedder`.

FlagEmbedding (and torch) are not installed in the offline test env, so the vendor SDK is
faked: a stub ``FlagEmbedding`` module is injected into ``sys.modules`` and the
``require_extra`` dependency gate is neutralized so construction proceeds. Tests stay fully
deterministic and offline (coding-standards §11).
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from indx.embed.base import Embedder
from indx.embed.bge_m3 import BGEM3Embedder
from indx.errors import MissingExtraError, StageError

if TYPE_CHECKING:
    from collections.abc import Callable

_DIM = 1024


class _FakeBGEM3FlagModel:
    """Stand-in for ``FlagEmbedding.BGEM3FlagModel``.

    Records construction args and returns deterministic dense vectors (one per input text,
    each of length ``_DIM``) shaped like FlagEmbedding's ``{"dense_vecs": ...}`` output.
    """

    last_init: dict[str, Any] = {}
    last_encode: dict[str, Any] = {}
    raise_on_encode = False

    def __init__(self, model_id: str, *, use_fp16: bool, devices: Any) -> None:
        type(self).last_init = {
            "model_id": model_id,
            "use_fp16": use_fp16,
            "devices": devices,
        }

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        type(self).last_encode = {"texts": list(texts), **kwargs}
        if type(self).raise_on_encode:
            raise RuntimeError("backend boom")
        # One length-_DIM vector per text; value derived from index for determinism.
        dense = [[float(i)] * _DIM for i, _ in enumerate(texts)]
        return {"dense_vecs": dense}


@pytest.fixture
def install_fake_flagembedding(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], type[_FakeBGEM3FlagModel]]:
    """Inject a fake ``FlagEmbedding`` module and neutralize the extra gate."""

    def _install() -> type[_FakeBGEM3FlagModel]:
        _FakeBGEM3FlagModel.raise_on_encode = False
        # Neutralize the extra gate. The adapter does `from ... import require_extra`,
        # so patch the name bound in the adapter's namespace (where it is looked up).
        monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
        monkeypatch.setattr("indx.embed.bge_m3.require_extra", lambda *a, **k: None)
        fake_module = types.ModuleType("FlagEmbedding")
        fake_module.BGEM3FlagModel = _FakeBGEM3FlagModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)
        return _FakeBGEM3FlagModel

    return _install


def test_satisfies_embedder_protocol(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    install_fake_flagembedding()
    emb = BGEM3Embedder()
    assert isinstance(emb, Embedder)
    assert emb.dim == _DIM


def test_records_actual_model_id_as_name(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    install_fake_flagembedding()
    emb = BGEM3Embedder(model_id="BAAI/bge-m3")
    # Instance name is the real model id (self-describing manifest).
    assert emb.name == "BAAI/bge-m3"


def test_embed_round_trips_dense_vectors(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    fake = install_fake_flagembedding()
    emb = BGEM3Embedder()
    texts = ["alpha", "beta", "gamma"]
    vectors = emb.embed(texts)
    # One vector per text, each of length dim, in input order.
    assert len(vectors) == len(texts)
    assert all(len(v) == emb.dim for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)
    assert vectors[1][0] == 1.0  # index-derived value confirms order preservation
    # The whole batch is handed to a single encode call (batching perf lever).
    assert fake.last_encode["texts"] == texts


def test_embed_empty_input_returns_empty_without_calling_backend(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    fake = install_fake_flagembedding()
    fake.last_encode = {}
    emb = BGEM3Embedder()
    assert emb.embed([]) == []
    assert fake.last_encode == {}  # backend not invoked


def test_default_use_fp16_off_for_determinism(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    fake = install_fake_flagembedding()
    BGEM3Embedder()
    assert fake.last_init["use_fp16"] is False


def test_backend_failure_wrapped_in_stage_error(
    install_fake_flagembedding: Callable[[], type[_FakeBGEM3FlagModel]],
) -> None:
    fake = install_fake_flagembedding()
    fake.raise_on_encode = True
    emb = BGEM3Embedder()
    with pytest.raises(StageError) as exc:
        emb.embed(["x"])
    assert "embed" in str(exc.value)


def test_construction_without_extra_raises_missing_extra() -> None:
    # No neutralization of require_extra: in the dep-absent env this must raise.
    with pytest.raises(MissingExtraError) as exc:
        BGEM3Embedder()
    assert "pip install indx[bge]" in str(exc.value)
