"""Unit tests for token-budgeted embedding batching (``indx.embed._batch``)."""

from __future__ import annotations

import pytest

from indx.embed._batch import (
    embed_in_batches,
    estimate_tokens,
    is_token_limit_error,
    token_budgeted_batches,
)


def _call(batches: list[list[str]]):
    """A fake embed call that records each batch and returns ``len(text)`` as the vector."""

    def call(batch: list[str]) -> list[list[float]]:
        batches.append(list(batch))
        return [[float(len(text))] for text in batch]

    return call


def test_estimate_tokens_counts_cjk_heavier_than_ascii() -> None:
    # 4 ASCII chars ≈ 1 token; 4 CJK chars ≈ 4 tokens. CJK must estimate higher.
    assert estimate_tokens("日本語テスト") > estimate_tokens("abcdef")
    assert estimate_tokens("") >= 1  # never zero, so empty inputs still advance the budget


def test_splits_on_item_count() -> None:
    batches = list(token_budgeted_batches(["a", "b", "c"], max_items=2, max_tokens=10_000))
    assert batches == [["a", "b"], ["c"]]


def test_splits_on_token_budget_even_under_item_cap() -> None:
    # Each 40-char ASCII text ≈ 11 tokens; a 25-token budget fits 2 per request despite a
    # generous 256-item cap. This is the CJK-overflow bug in miniature.
    texts = ["x" * 40, "y" * 40, "z" * 40]
    batches = list(token_budgeted_batches(texts, max_items=256, max_tokens=25))
    assert [len(b) for b in batches] == [2, 1]


def test_single_oversized_text_is_yielded_alone() -> None:
    texts = ["small", "B" * 10_000, "small"]
    batches = list(token_budgeted_batches(texts, max_items=256, max_tokens=100))
    assert batches[1] == ["B" * 10_000]  # cannot be split further; goes out on its own


def test_embed_in_batches_preserves_order_and_count() -> None:
    seen: list[list[str]] = []
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = embed_in_batches(texts, max_items=2, max_tokens=10_000, call=_call(seen))
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(b) for b in seen] == [2, 2, 1]


def test_is_token_limit_error_matches_provider_message() -> None:
    msg = "Error code: 400 - Requested 342105 tokens, max 300000 tokens per request"
    assert is_token_limit_error(RuntimeError(msg))
    assert is_token_limit_error(RuntimeError("max_tokens_per_request"))
    assert not is_token_limit_error(RuntimeError("rate limit exceeded"))


def test_bisects_and_retries_when_provider_rejects_token_cap() -> None:
    """A batch the estimate let through but the provider rejects is halved and retried."""
    seen: list[list[str]] = []

    def call(batch: list[str]) -> list[list[float]]:
        # Reject any request of more than 2 items as if it overflowed the token cap.
        if len(batch) > 2:
            raise RuntimeError("Requested 999 tokens, max 300000 tokens per request")
        seen.append(list(batch))
        return [[float(len(t))] for t in batch]

    texts = ["a", "bb", "ccc", "dddd"]
    # max_tokens is huge so the pre-split keeps all 4 together; only the adaptive retry saves us.
    vectors = embed_in_batches(texts, max_items=256, max_tokens=10_000, call=call)
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0]  # order preserved across the split
    assert all(len(b) <= 2 for b in seen)


def test_non_token_errors_propagate_without_retry() -> None:
    calls = {"n": 0}

    def call(batch: list[str]) -> list[list[float]]:
        calls["n"] += 1
        raise RuntimeError("authentication failed")

    with pytest.raises(RuntimeError, match="authentication"):
        embed_in_batches(["a", "b"], max_items=256, max_tokens=10_000, call=call)
    assert calls["n"] == 1  # not retried — only token-cap errors trigger the bisect
