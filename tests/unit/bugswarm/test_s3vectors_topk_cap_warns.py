"""Regression: S3VectorsStore.search must not silently truncate k > 100.

AWS QueryVectors hard-caps ``topK`` at 100 with no pagination, so the adapter cannot return
more than 100 hits. The contract gap was that it clamped the request silently. This test pins
the now-surfaced behavior: a request for ``k > 100`` is clamped to 100 (the service maximum) AND
a WARNING is logged so the truncation is never silent; a request for ``k <= 100`` passes through
untouched and logs nothing.

boto3 is absent here, so a fake ``s3vectors`` client is injected (mirrors
``tests/unit/store/test_s3vectors.py``).
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from indx.store.s3vectors import _MAX_TOPK, S3VectorsStore


class _CapturingClient:
    """Records the ``topK`` each query_vectors call received; returns no hits."""

    def __init__(self) -> None:
        self.top_ks: list[int] = []
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def query_vectors(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        queryVector: dict[str, list[float]],
        topK: int,
        returnDistance: bool,
        returnMetadata: bool,
    ) -> dict[str, Any]:
        self.top_ks.append(topK)
        return {"vectors": []}


@pytest.fixture(autouse=True)
def _restore_indx_log_propagation() -> Any:
    """``indx.utils.logging.setup_logging`` sets ``propagate=False`` on the ``indx`` logger;
    if an earlier test triggered it, ``caplog`` (which captures at the root) would miss these
    records even though the warning is emitted. Force propagation for the duration of the test."""
    lg = logging.getLogger("indx")
    prev = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = prev


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> S3VectorsStore:
    monkeypatch.setattr("indx.store.s3vectors.require_extra", lambda *a, **k: None)
    client = _CapturingClient()
    module = types.ModuleType("boto3")
    module.client = lambda service, **kwargs: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", module)
    s = S3VectorsStore(bucket="b", dim=2)
    s._client = client  # type: ignore[assignment]
    return s


def _capturing(store: S3VectorsStore) -> _CapturingClient:
    return store._client  # type: ignore[return-value]


def test_search_clamps_oversized_k_to_service_max_and_warns(
    store: S3VectorsStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="indx.store.s3vectors"):
        store.search([1.0, 0.0], k=250)
    # The service ceiling is enforced: at most _MAX_TOPK was requested from AWS.
    assert _capturing(store).top_ks == [_MAX_TOPK]
    # The truncation is surfaced, not silent.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("250" in r.getMessage() and str(_MAX_TOPK) in r.getMessage() for r in warnings)


def test_search_within_cap_passes_k_through_without_warning(
    store: S3VectorsStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="indx.store.s3vectors"):
        store.search([1.0, 0.0], k=42)
    assert _capturing(store).top_ks == [42]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_search_at_exact_cap_does_not_warn(
    store: S3VectorsStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="indx.store.s3vectors"):
        store.search([1.0, 0.0], k=_MAX_TOPK)
    assert _capturing(store).top_ks == [_MAX_TOPK]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_close_releases_client_and_is_idempotent(store: S3VectorsStore) -> None:
    client = _capturing(store)
    store.close()
    assert client.close_calls == 1
    # After close the client reference is dropped, and a second close is a safe no-op.
    store.close()
    assert client.close_calls == 1
