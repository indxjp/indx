"""T2 — retrieval mechanics over a real corpus (US-1/US-2, testing-strategy §3.7).

Asserts the *mechanics*, not semantic quality (the hash embedder is non-semantic): hits
come back in score order, carry source-document lineage, and the no-DB store is consistent
with the embeddings stored in the space.
"""

from __future__ import annotations

import pytest

from indx.store.jsonl import JsonlStore

pytestmark = pytest.mark.corpus


def test_query_returns_ranked_hits_with_lineage(build_corpus) -> None:
    space = build_corpus("nested_handbook")
    store = JsonlStore()
    store.upsert(space.chunks)

    # Use a real chunk's own text as the query → it should rank itself at/near the top.
    target = next(c for c in space.chunks if "security" in c.text.lower())
    from indx.embed.hash_embedder import HashEmbedder

    vec = HashEmbedder().embed([target.text])[0]
    hits = store.search(vec, k=3)

    assert hits, "expected hits"
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert hits[0].chunk.id == target.id  # self-retrieval is the top hit
    # the hit's document is resolvable and carries the source doc's folder lineage (US-2)
    doc = space.document(hits[0].chunk.doc_id)
    assert doc is not None
    assert doc.id == target.doc_id
    assert doc.lineage == space.document(target.doc_id).lineage and doc.lineage
