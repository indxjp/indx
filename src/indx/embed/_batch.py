"""Token-budgeted batching for embedding adapters.

Embedding APIs cap a request by two independent limits: the number of inputs *and* the
total tokens across them (OpenAI: 300k tokens/request). Splitting by item count alone
overflows the token cap on dense text — Japanese and other CJK content packs far more
tokens per item than English, so a 256-item batch that is comfortable for English prose
can blow past 300k tokens (the ``max_tokens_per_request`` 400 we are fixing).

This module sub-splits each item-count batch to keep an *estimated* token total under
budget, then bisects-and-retries any batch the provider still rejects for the token cap —
so correctness never depends on the estimate being exact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

# OpenAI caps embeddings at 300k tokens/request. Default a margin below it so estimation
# slack (see ``estimate_tokens``) does not push a batch over the real ceiling.
MAX_TOKENS_PER_REQUEST = 290_000


def estimate_tokens(text: str) -> int:
    """Conservative token estimate that needs no tokenizer.

    Non-ASCII characters (CJK, etc.) are byte-level BPE'd into roughly one-or-more tokens
    each, so they are counted at full weight; ASCII packs at ~4 chars/token. The estimate
    is deliberately biased to *over*-count, which only yields more, smaller requests — never
    a request over the provider's token cap. The bisect-on-error fallback in
    :func:`embed_in_batches` covers the rare case where it still under-counts.
    """
    ascii_chars = sum(1 for ch in text if ch.isascii())
    wide_chars = len(text) - ascii_chars
    return wide_chars + (ascii_chars + 3) // 4 + 1


def token_budgeted_batches(
    texts: Sequence[str],
    *,
    max_items: int,
    max_tokens: int,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> Iterator[list[str]]:
    """Yield contiguous slices of ``texts`` bounded by both item count and token budget.

    A single text whose own estimate exceeds ``max_tokens`` is yielded alone — it cannot be
    split further, and the provider enforces its own per-input token limit independently.
    """
    batch: list[str] = []
    tokens = 0
    for text in texts:
        t = count_tokens(text)
        if batch and (len(batch) >= max_items or tokens + t > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += t
    if batch:
        yield batch


def is_token_limit_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a provider's per-request token-cap rejection."""
    msg = str(exc).lower()
    return "max_tokens_per_request" in msg or "tokens per request" in msg


def _embed_batch_adaptive(
    batch: list[str], call: Callable[[list[str]], list[list[float]]]
) -> list[list[float]]:
    """Embed one batch, halving and retrying if the provider rejects it on the token cap."""
    try:
        return call(batch)
    except Exception as exc:  # noqa: BLE001 — inspected, then re-raised if not a token-cap error
        if len(batch) > 1 and is_token_limit_error(exc):
            mid = len(batch) // 2
            return _embed_batch_adaptive(batch[:mid], call) + _embed_batch_adaptive(
                batch[mid:], call
            )
        raise


def embed_in_batches(
    texts: Sequence[str],
    *,
    max_items: int,
    max_tokens: int,
    call: Callable[[list[str]], list[list[float]]],
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[list[float]]:
    """Embed ``texts`` in token-budgeted batches, returning one vector per input in order.

    ``call(batch)`` performs one provider request and must return vectors in the batch's
    input order. Batches are pre-split to stay under ``max_tokens``; any batch the provider
    still rejects for the token cap is bisected and retried, so a stray estimate miss costs
    an extra round-trip rather than a failed run.
    """
    vectors: list[list[float]] = []
    for batch in token_budgeted_batches(
        texts, max_items=max_items, max_tokens=max_tokens, count_tokens=count_tokens
    ):
        vectors.extend(_embed_batch_adaptive(batch, call))
    return vectors
