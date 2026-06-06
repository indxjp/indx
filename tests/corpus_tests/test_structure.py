"""T2 — folder lineage is recovered from real corpus directories (FR-S01, US-2)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("corpus", ["nested_handbook", "split_report"])
def test_lineage_matches_labels(corpus, build_corpus, load_labels) -> None:
    space = build_corpus(corpus)
    labels = load_labels(corpus)
    by_path = {d.path: d for d in space.documents_}

    assert set(by_path) == set(labels["lineage"]), "document set differs from labels"
    for path, expected_lineage in labels["lineage"].items():
        assert by_path[path].lineage == expected_lineage


def test_all_documents_chunked_and_embedded(build_corpus) -> None:
    space = build_corpus("nested_handbook")
    assert space.chunks, "expected at least one chunk"
    assert all(c.embedding is not None for c in space.chunks)
    # every document produced at least one chunk
    chunked_docs = {c.doc_id for c in space.chunks}
    assert chunked_docs == {d.id for d in space.documents_}
