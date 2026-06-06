"""T1 — .indx archive write → read round-trip and reproducibility."""

from __future__ import annotations

import pytest

from indx import DirectoryPipeline
from indx.archive import read_archive, write_archive
from indx.errors import ArchiveError


def _build(tmp_tree):
    # Zero-dependency offline core stack (the cloud defaults need OPENAI_API_KEY).
    return DirectoryPipeline(parser="plaintext", llm="none", embedder="hash", store="jsonl").run(
        tmp_tree
    )


def test_roundtrip_preserves_graph(tmp_tree, tmp_path) -> None:
    space = _build(tmp_tree)
    dest = tmp_path / "s.indx"
    write_archive(space, dest)
    loaded = read_archive(dest)
    assert [d.path for d in loaded.documents_] == [d.path for d in space.documents_]
    assert len(loaded.chunks) == len(space.chunks)
    assert [r.model_dump(mode="json") for r in loaded.relations] == [
        r.model_dump(mode="json") for r in space.relations
    ]


def test_archive_is_byte_reproducible(tmp_tree, tmp_path) -> None:
    space = _build(tmp_tree)
    a, b = tmp_path / "a.indx", tmp_path / "b.indx"
    write_archive(space, a)
    write_archive(space, b)
    assert a.read_bytes() == b.read_bytes()  # determinism guardrail (NFR-DET-1)


def test_reader_rejects_non_archive(tmp_path) -> None:
    bogus = tmp_path / "x.indx"
    bogus.write_text("not a zip")
    with pytest.raises(ArchiveError):
        read_archive(bogus)
