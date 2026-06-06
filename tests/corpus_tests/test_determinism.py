"""T2 — deterministic outputs on real corpora (NFR-DET-1, testing-strategy §3.5).

Builds each corpus twice and asserts the deterministic projection (structure + relations +
archive bytes) is identical. Stochastic enrichment is excluded by construction — the default
enricher is deterministic, and a real LLM enricher would be asserted via cassette instead.
"""

from __future__ import annotations

import pytest

from indx.archive.writer import write_archive

pytestmark = pytest.mark.corpus


def _structure(space) -> dict:
    return {
        "documents": [(d.path, d.lineage) for d in space.documents_],
        "chunks": [(c.doc_id, c.position, c.text) for c in space.chunks],
        "relations": sorted((r.type.value, r.src, r.dst) for r in space.relations),
    }


@pytest.mark.parametrize("corpus", ["nested_handbook", "split_report", "airgap_smoke"])
def test_structure_is_stable_across_runs(corpus, build_corpus) -> None:
    assert _structure(build_corpus(corpus)) == _structure(build_corpus(corpus))


def test_archive_bytes_are_reproducible(tmp_path, build_corpus) -> None:
    space = build_corpus("nested_handbook")
    a, b = tmp_path / "a.indx", tmp_path / "b.indx"
    write_archive(space, a)
    write_archive(space, b)
    assert a.read_bytes() == b.read_bytes()
