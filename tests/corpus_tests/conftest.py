"""Harness for the corpus tier (testing-strategy §3.2/§3.6).

Provides:
- ``build_corpus``     — run the real (offline, zero-dep) pipeline on a corpus directory.
- ``load_labels``      — parse a corpus' hand-authored ``*.labels.yaml``.
- ``relation_prr``     — precision/recall of inferred relations vs. labels, per type.

The default pipeline (plaintext + hash + jsonl) is deterministic and offline, so corpus
tests need no recorded ParsedDoc replay yet. When a heavy parser becomes the default, swap
``build_corpus`` to feed recorded ``_recorded/parsed/*`` instead (testing-strategy §3.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml  # provided by the test env (pyyaml); see noxfile dev deps

from indx import DirectoryPipeline, KnowledgeSpace, RelationType

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"


def corpus_dir(name: str) -> Path:
    return CORPUS_ROOT / name


@pytest.fixture
def build_corpus():
    def _build(name: str) -> KnowledgeSpace:
        # Zero-dependency offline core stack (the cloud defaults need OPENAI_API_KEY).
        return DirectoryPipeline(
            seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl"
        ).run(corpus_dir(name))

    return _build


@pytest.fixture
def load_labels():
    def _load(name: str) -> dict:
        # Sidecar lives next to the corpus dir, so the corpus dir stays pure documents.
        path = CORPUS_ROOT / f"{name}.labels.yaml"
        return yaml.safe_load(path.read_text())

    return _load


@dataclass
class Score:
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def _path_of(space: KnowledgeSpace, doc_id: str) -> str:
    doc = space.document(doc_id)
    return doc.path if doc else doc_id


def _relation_prr(
    space: KnowledgeSpace,
    rel_type: RelationType,
    expected_pairs: list[tuple[str, str]],
    *,
    directed: bool,
) -> Score:
    """Precision/recall for one relation type, comparing (src_path, dst_path) pairs."""

    def norm(a: str, b: str) -> tuple[str, str]:
        return (a, b) if directed else tuple(sorted((a, b)))  # type: ignore[return-value]

    got = {
        norm(_path_of(space, r.src), _path_of(space, r.dst))
        for r in space.relations
        if r.type == rel_type
    }
    want = {norm(a, b) for a, b in expected_pairs}
    tp = len(got & want)
    fp = len(got - want)
    fn = len(want - got)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return Score(precision, recall, tp, fp, fn)


@pytest.fixture
def relation_prr():
    """Expose the scorer as a fixture so corpus tests need no cross-module import."""
    return _relation_prr
