"""Extras e2e for the optional embedded STORE backends, against their real libraries.

Chroma and LanceDB are *embedded* (no server, no network, no key): they persist to a local
directory. This is the same upsert → search → delete conformance bar the integration
``test_store_contract`` holds Qdrant to, run here against the real embedded libraries so a
build that selects ``--store chroma`` / ``--store lancedb`` is proven end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.core.chunk import Chunk
from indx.store.base import VectorStore

pytestmark = pytest.mark.extras


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(id=cid, doc_id="d", position=0, text=cid, embedding=vec)


def _exercise_store(store: VectorStore) -> None:
    store.upsert([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0]), _chunk("c", [1.0, 1.0])])

    hits = store.search([1.0, 0.0], k=2)
    assert [h.chunk.id for h in hits][0] == "a", "nearest neighbour should rank first"
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)), (
        "hits must be in descending score order"
    )

    store.delete(["a"])
    after = store.search([1.0, 0.0], k=3)
    assert "a" not in [h.chunk.id for h in after], "deleted chunk must not resurface"


def test_chroma_round_trip(requires_lib, tmp_path: Path) -> None:
    requires_lib("chromadb")
    from indx.store.chroma import ChromaStore

    _exercise_store(ChromaStore(path=str(tmp_path / "chroma")))


def test_lancedb_round_trip(requires_lib, tmp_path: Path) -> None:
    requires_lib("lancedb")
    from indx.store.lancedb import LanceDBStore

    _exercise_store(LanceDBStore(path=str(tmp_path / "space.lancedb")))
