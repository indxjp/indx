"""T2 — relation inference quality on real corpora vs. hand labels (testing-strategy §3.6).

This is the R-3 guard: relations are scored with precision/recall against the labels, and
the per-corpus floors are asserted. Lowering a floor is a deliberate, reviewable act.
"""

from __future__ import annotations

import pytest

from indx import RelationType

pytestmark = pytest.mark.corpus


def test_nested_handbook_sibling_and_parent(build_corpus, load_labels, relation_prr) -> None:
    space = build_corpus("nested_handbook")
    labels = load_labels("nested_handbook")

    sib = relation_prr(space, RelationType.SIBLING, labels["sibling_pairs"], directed=False)
    assert sib.precision >= labels["floors"]["sibling"]["precision"]
    assert sib.recall >= labels["floors"]["sibling"]["recall"]

    par = relation_prr(space, RelationType.PARENT, labels["parent_edges"], directed=True)
    assert par.precision >= labels["floors"]["parent"]["precision"]
    assert par.recall >= labels["floors"]["parent"]["recall"]


def test_split_report_continues_chain(build_corpus, load_labels, relation_prr) -> None:
    space = build_corpus("split_report")
    labels = load_labels("split_report")
    cont = relation_prr(space, RelationType.CONTINUES, labels["continues_chain"], directed=True)
    assert cont.precision >= labels["floors"]["continues"]["precision"]
    assert cont.recall >= labels["floors"]["continues"]["recall"]
