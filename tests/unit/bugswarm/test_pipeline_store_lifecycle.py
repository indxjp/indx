"""Regression tests for four confirmed DirectoryPipeline defects (offline, mock-based).

Covered findings (all in ``src/indx/pipeline/pipeline.py``, plus the resource-leak contract
in ``src/indx/store/base.py``):

1. Cache-write failure must NOT misclassify a cleanly-parsed doc as a parse skip. The
   ``StageCache.put`` write is a best-effort side channel: an ``OSError`` there (disk full,
   read-only ``--out``) must leave the parsed document in ``ctx.parsed``, never in ``ctx.errors``.
2. ``use()`` must re-forward the user's pinned ``[store.<backend>]`` options when it rebuilds a
   dim-sized store (rebinding the embedder, or the store by name), instead of dropping them.
3. The ``embedder`` property is honestly typed ``Embedder | None`` and returns ``None`` when the
   pipeline was built with ``embed=False`` (no ``AttributeError`` masked behind a cast).
4. The pipeline ``close()`` (and context-manager) releases a registry-built store via the new
   ``VectorStore.close`` contract, but never touches a user-supplied store.

Style mirrors ``tests/unit/test_pipeline_dim.py`` (spy ``get_store``, fake embedder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from indx.config import Config
from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.parsed import Block, ParsedDoc
from indx.pipeline import pipeline as pipeline_mod
from indx.pipeline.pipeline import DirectoryPipeline, ResumableParseStage
from indx.store.base import SearchHit, VectorStore, close_store
from indx.utils.cache import StageCache


def _config_with_store_url(url: str) -> Config:
    """A Config pinning ``[store.qdrant] url`` so use() rebuilds can be checked for drops."""
    return Config.model_validate({"store": {"backend": "qdrant", "qdrant": {"url": url}}})


class _FakeEmbedder:
    """Embedder protocol stand-in with a fixed, declared dimensionality."""

    def __init__(self, *, name: str = "fake", dim: int = 1536) -> None:
        self.name = name
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class _SpyStore:
    """Records constructor kwargs and tracks whether ``close()`` was called."""

    def __init__(self, **kwargs: Any) -> None:
        self.name = "qdrant"
        self.kwargs = kwargs
        self.closed = False

    def upsert(self, chunks: list[Any]) -> None:  # pragma: no cover - not exercised
        return None

    def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:  # pragma: no cover
        return []

    def delete(self, chunk_ids: list[str]) -> None:  # pragma: no cover
        return None

    def close(self) -> None:
        self.closed = True


def _spy_get_store(monkeypatch: pytest.MonkeyPatch) -> list[_SpyStore]:
    """Patch ``get_store`` in the pipeline module; return every store it builds, in order."""
    built: list[_SpyStore] = []

    def fake_get_store(name: str, **kwargs: Any) -> _SpyStore:
        store = _SpyStore(**kwargs)
        built.append(store)
        return store

    monkeypatch.setattr(pipeline_mod, "get_store", fake_get_store)
    return built


# ---------------------------------------------------------------- finding 1: cache-write skip


class _OkParser:
    """Always parses successfully into a one-block ParsedDoc."""

    name = "plaintext"

    def parse(self, path: Path) -> ParsedDoc:
        return ParsedDoc(source_path=str(path), blocks=[Block(text="hello", order=0)])


def _ctx_with_one_doc(tmp_path: Path) -> SpaceContext:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    ctx = SpaceContext(root=tmp_path)
    ctx.space.documents_.append(Document(id="d1", path="a.txt"))
    return ctx


def test_cache_write_failure_keeps_parsed_doc(tmp_path: Path) -> None:
    # A successfully parsed doc whose cache ``put`` raises OSError must still land in
    # ctx.parsed (not be recorded as a kind="skip" parse error and dropped).
    cache = StageCache(tmp_path, enabled=True)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full / read-only --out")

    cache.put = boom  # type: ignore[method-assign]
    stage = ResumableParseStage(_OkParser(), "plaintext", cache, jobs=1)

    ctx = stage.run(_ctx_with_one_doc(tmp_path))

    assert "d1" in ctx.parsed
    assert ctx.parsed["d1"].text == "hello"
    assert not any(e.kind == "skip" for e in ctx.errors)


def test_real_parse_failure_still_recorded_as_skip(tmp_path: Path) -> None:
    # The fix must not swallow genuine parse failures — those stay kind="skip".
    class _BadParser:
        name = "plaintext"

        def parse(self, path: Path) -> ParsedDoc:
            raise ValueError("cannot parse")

    cache = StageCache(tmp_path, enabled=True)
    stage = ResumableParseStage(_BadParser(), "plaintext", cache, jobs=1)

    ctx = stage.run(_ctx_with_one_doc(tmp_path))

    assert "d1" not in ctx.parsed
    skips = [e for e in ctx.errors if e.kind == "skip"]
    assert len(skips) == 1
    assert skips[0].item == "a.txt"


# ---------------------------------------------------------------- finding 2: use() drops opts


def test_use_rebind_embedder_preserves_store_options(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _spy_get_store(monkeypatch)
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",  # dim 256
        store="qdrant",
        output="jsonl",
        config=_config_with_store_url("http://prod:6333"),
    )
    assert built[0].kwargs.get("url") == "http://prod:6333"

    # Rebinding only the embedder re-sizes the dim-sized store; the pinned url must survive.
    pipe.use(embedder=_FakeEmbedder(dim=1024))
    rebuilt = built[-1]
    assert rebuilt.kwargs.get("url") == "http://prod:6333"
    assert rebuilt.kwargs.get("dim") == 1024


def test_use_rebind_store_by_name_preserves_options(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _spy_get_store(monkeypatch)
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
        config=_config_with_store_url("http://prod:6333"),
    )
    pipe.use(store="qdrant")
    assert built[-1].kwargs.get("url") == "http://prod:6333"


# ---------------------------------------------------------------- finding 3: embedder property


def test_embedder_property_is_none_when_embed_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _spy_get_store(monkeypatch)
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
        embed=False,
    )
    # Honest Optional: None rather than a cast that masks an AttributeError on first access.
    assert pipe.embedder is None


# ---------------------------------------------------------------- finding 4: store close()


def test_close_releases_registry_built_store(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _spy_get_store(monkeypatch)
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
    )
    store = built[0]
    pipe.close()
    assert store.closed is True
    pipe.close()  # idempotent — no error on a second call


def test_close_leaves_user_supplied_store_alone() -> None:
    user_store = _SpyStore()
    pipe = DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store=user_store,
        output="jsonl",
    )
    pipe.close()
    assert user_store.closed is False


def test_context_manager_closes_store(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _spy_get_store(monkeypatch)
    with DirectoryPipeline(
        parser="plaintext",
        llm="none",
        embedder="hash",
        store="qdrant",
        output="jsonl",
    ):
        pass
    assert built[0].closed is True


def test_close_store_helper_is_noop_without_close() -> None:
    # close_store() is best-effort: a backend with no close() must still satisfy the
    # VectorStore protocol (isinstance) AND be safe to pass to close_store() (no crash).
    class _Minimal:
        name = "mini"

        def upsert(self, chunks: list[Any]) -> None:
            return None

        def search(self, vector: list[float], k: int = 5) -> list[SearchHit]:
            return []

        def delete(self, chunk_ids: list[str]) -> None:
            return None

    store = _Minimal()
    assert isinstance(store, VectorStore)  # close() stays out of the runtime-checkable surface
    close_store(store)  # no-op, must not raise


def test_close_store_helper_invokes_close_when_present() -> None:
    spy = _SpyStore()
    close_store(spy)
    assert spy.closed is True
