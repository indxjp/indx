"""T3 — the shared store conformance contract, run against a REAL service (testing-strategy §4.4d).

Every VectorStore must clear the same bar: upsert → search returns top-k in descending
order → delete removes. The same parametrized body runs against the in-core JsonlStore (as
a baseline) and, when a service URL is present, the real Qdrant store. This is the contract
test that guards adapters against ecosystem churn (R-4).
"""

from __future__ import annotations

import os

import pytest

from indx.core.chunk import Chunk
from indx.store.jsonl import JsonlStore

pytestmark = pytest.mark.integration


def _make_store(kind: str):
    if kind == "jsonl":
        return JsonlStore()
    if kind == "qdrant":
        if not os.environ.get("INDX_QDRANT_URL"):
            pytest.skip("INDX_QDRANT_URL not set; real Qdrant service absent")
        from indx.registry import get_store  # deferred: needs the qdrant extra

        return get_store("qdrant", url=os.environ["INDX_QDRANT_URL"])
    raise AssertionError(kind)


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)


@pytest.mark.parametrize("kind", ["jsonl", "qdrant"])
def test_store_conformance(kind: str) -> None:
    store = _make_store(kind)
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk.id == "a"

    store.delete(["a"])
    remaining = {h.chunk.id for h in store.search([1.0, 0.0], k=5)}
    assert "a" not in remaining
