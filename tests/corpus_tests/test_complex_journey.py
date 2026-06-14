"""T2/corpus — headline end-to-end journey on the complex acme_kb fixture.

This is the proof that indx holds up on a realistic, deeply-nested, mixed-type corpus
(24 .md/.txt/.rst docs across handbook/engineering/finance/policies). It exercises the
full offline stack end-to-end: structure + lineage, relation inference quality, lexical
retrieval, the CLI build/inspect/query roundtrip, .zip-input equivalence, the agent
connector surface (search/overview/get_document/dispatch + tool schemas), and determinism.
Everything is deterministic and offline (plaintext + hash + jsonl, seed=0), so this tier
runs green in CI by default.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from indx import DirectoryPipeline, RelationType
from indx.agent import connect

pytestmark = pytest.mark.corpus

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
ACME_DIR = CORPUS_ROOT / "acme_kb"


def _offline_pipeline() -> DirectoryPipeline:
    return DirectoryPipeline(seed=0, parser="plaintext", llm="none", embedder="hash", store="jsonl")


def test_structure_matches_labels(build_corpus, load_labels) -> None:
    space = build_corpus("acme_kb")
    labels = load_labels("acme_kb")

    assert space.stats.documents == labels["documents"]

    by_path = {d.path: d for d in space.documents_}
    assert set(by_path) == set(labels["lineage"])
    for path, expected_lineage in labels["lineage"].items():
        assert by_path[path].lineage == expected_lineage

    assert space.chunks, "expected at least one chunk"
    assert all(c.embedding is not None for c in space.chunks)

    chunked_docs = {c.doc_id for c in space.chunks}
    assert chunked_docs == {d.id for d in space.documents_}

    assert space.manifest.embedding_model == "hash"
    assert dict(space.stats.types) == labels["types"]


def test_relations_match_labels(build_corpus, load_labels, relation_prr) -> None:
    space = build_corpus("acme_kb")
    labels = load_labels("acme_kb")

    sib = relation_prr(space, RelationType.SIBLING, labels["sibling_pairs"], directed=False)
    assert sib.precision == 1.0
    assert sib.recall == 1.0

    par = relation_prr(space, RelationType.PARENT, labels["parent_edges"], directed=True)
    assert par.precision == 1.0
    assert par.recall == 1.0

    cont = relation_prr(space, RelationType.CONTINUES, labels["continues_chain"], directed=True)
    assert cont.precision == 1.0
    assert cont.recall == 1.0

    references = [r for r in space.relations if r.type == RelationType.REFERENCES]
    assert len(references) >= labels["floors"]["references_min"]


def test_retrieval_top_hit_matches_labels(build_corpus, load_labels) -> None:
    space = build_corpus("acme_kb")
    labels = load_labels("acme_kb")

    for entry in labels["retrieval"]:
        hits = space.search(entry["query"], k=1)
        assert hits, f"no hits for query: {entry['query']!r}"
        top = hits[0]
        assert top.source is not None
        assert top.source.path == entry["top"]
        assert top.score > 0


def test_cli_build_inspect_query_roundtrip(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from indx.cli.app import app

    runner = CliRunner()
    out = tmp_path / "acme.indx"

    build = runner.invoke(
        app,
        ["build", str(ACME_DIR), "--out", str(out), "--offline", "--name", "acme"],
    )
    assert build.exit_code == 0, build.output
    assert out.exists()

    inspect = runner.invoke(app, ["inspect", str(out)])
    assert inspect.exit_code == 0, inspect.output
    assert "documents=24" in inspect.stdout

    query = runner.invoke(app, ["query", "retirement 401k matching", str(out), "-k", "2"])
    assert query.exit_code == 0, query.output
    assert query.stdout.strip()


def test_zip_input_equivalence(build_corpus, tmp_path: Path) -> None:
    zip_path = tmp_path / "acme_kb.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(ACME_DIR.rglob("*")):
            if f.is_file():
                # arcname relative to the corpus root so files sit at the zip root.
                zf.write(f, arcname=str(f.relative_to(ACME_DIR)))

    space = _offline_pipeline().run(zip_path)
    assert space.stats.documents == 24

    zip_paths = {d.path for d in space.documents_}
    dir_paths = {d.path for d in build_corpus("acme_kb").documents_}
    assert zip_paths == dir_paths


def test_connector_journey(build_corpus, load_labels) -> None:
    labels = load_labels("acme_kb")
    space = build_corpus("acme_kb")
    kb = connect(space, name="acme")

    ov = kb.overview(sample=3)
    assert ov.documents == 24
    assert ov.chunks == 24
    assert ov.embedding_model == "hash"
    assert ov.embedding_dim == 256
    assert ov.types == labels["types"]
    assert len(ov.sample_documents) == 3

    sr = kb.search("retirement 401k matching contribution dollar", k=2)
    assert sr.count >= 1
    assert sr.hits[0].source.endswith("retirement-401k.md")
    assert sr.hits[0].score > 0
    scores = [h.score for h in sr.hits]
    assert scores == sorted(scores, reverse=True)

    typed = kb.search("two-factor authentication hardware key", k=5, doc_type="policy")
    assert typed.hits, "expected at least one policy hit"
    for hit in typed.hits:
        assert hit.doc_type == "policy"
        assert hit.source.startswith("policies/")

    doc = kb.get_document("vector-store.md")
    assert doc is not None
    assert doc.path == "engineering/architecture/vector-store.md"
    assert doc.chunk_count >= 1
    assert doc.text

    real_id = space.documents_[0].id
    assert kb.get_document(real_id) is not None
    assert kb.get_document("does-not-exist.md") is None

    result = kb.call("indx_search", {"query": "incident on-call severity", "k": 1})
    assert set(result.keys()) == {"query", "count", "hits"}
    assert result["hits"][0]["source"].endswith("incident-response.md")
    with pytest.raises(ValueError):
        kb.call("not_a_real_tool", {})

    assert [t.name for t in kb.tools()] == [
        "indx_search",
        "indx_overview",
        "indx_get_document",
    ]
    assert kb.openai_schema()[0]["type"] == "function"
    anthropic = kb.anthropic_schema()[0]
    assert {"name", "description", "input_schema"} <= set(anthropic)


def test_determinism(build_corpus) -> None:
    first = build_corpus("acme_kb")
    second = build_corpus("acme_kb")

    assert sorted(d.path for d in first.documents_) == sorted(d.path for d in second.documents_)
    assert len(first.chunks) == len(second.chunks)
    assert len(first.relations) == len(second.relations)
