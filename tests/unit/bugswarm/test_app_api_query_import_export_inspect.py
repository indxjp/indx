"""Regression tests for ``indx.app.api`` defects fixed by the bug swarm.

Covers four independent fixes in :mod:`indx.app.api`:

* ``POST /api/query`` clamps a negative/zero ``k`` to ``1`` instead of silently slicing the hit
  list from the end (``hits[:-2]``), matching the agent connector's ``max(int(k), 1)`` clamp.
* ``POST /api/import`` gives each upload a unique basename, so two same-named uploads can't
  overwrite each other (the first caller's returned path pointing at the second caller's bytes).
* ``GET /api/export`` refuses a bare file that is not an ``.indx`` artifact (e.g. ``indx.toml``),
  matching the directory branch's ``*.indx`` constraint.
* ``GET /api/inspect``/``POST /api/query`` return 400 (not an unhandled 500) when a jsonl output
  directory has a present-but-malformed ``manifest.json``.

The whole module self-skips when the ``app`` extra (fastapi) is not installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from indx.app.server import create_app  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "notes").mkdir(parents=True)
    (src / "notes" / "alpha.md").write_text("# Alpha\n\nThe onboarding handbook lives here.\n")
    (src / "beta.md").write_text("# Beta\n\nSecurity policy details.\n")
    return src


def _build_space(client: TestClient, corpus: Path, out: Path) -> None:
    """Run an offline SSE build into ``out`` and assert it completed."""
    body = {"directory": str(corpus), "out": str(out), "offline": True}
    events: list[str] = []
    current = ""
    with client.stream("POST", "/api/build", json=body) as resp:
        assert resp.status_code == 200, resp.read()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append(current)
    assert "done" in events, events


def test_query_clamps_negative_k_instead_of_tail_slice(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    """A negative ``k`` must not silently drop the top hits via ``hits[:-k]``.

    Before the fix ``k=-1`` reached ``hits[:-1]`` (and the ``len(kept) >= k`` loop guard never
    tripped), returning a wrong tail-truncated slice. After the clamp ``k`` behaves like ``1``.
    """
    out = tmp_path / "ks"
    _build_space(client, corpus, out)

    resp = client.post("/api/query", json={"space": str(out), "text": "handbook", "k": -1})
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]
    # Clamped to 1: a positive, bounded result rather than an end-truncated slice.
    assert len(hits) == 1


def test_import_unique_name_no_overwrite(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two same-named uploads land on distinct paths; neither clobbers the other's bytes."""
    pytest.importorskip("multipart")
    from indx.app import api

    workdir = tmp_path / "import-work"
    monkeypatch.setenv("INDX_APP_IMPORT_DIR", str(workdir))
    api._import_workdir.cache_clear()
    try:
        first = client.post(
            "/api/import",
            files={"file": ("corpus.zip", b"first-bytes", "application/zip")},
        )
        second = client.post(
            "/api/import",
            files={"file": ("corpus.zip", b"second-bytes", "application/zip")},
        )
    finally:
        api._import_workdir.cache_clear()

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    p1 = Path(first.json()["path"])
    p2 = Path(second.json()["path"])
    assert p1 != p2, "same-named uploads must not collide on one target"
    # Each file still holds its own caller's bytes (no overwrite / no interleave).
    assert p1.read_bytes() == b"first-bytes"
    assert p2.read_bytes() == b"second-bytes"


def test_export_rejects_non_indx_file(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readable non-``.indx`` file under an allowed root is rejected (400), not streamed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "indx.toml").write_text("[indx]\nname = 'secret'\n")
    resp = client.get("/api/export", params={"space": "indx.toml"})
    assert resp.status_code == 400, resp.text


def test_inspect_malformed_manifest_returns_400(client: TestClient, tmp_path: Path) -> None:
    """A jsonl dir with a present-but-malformed ``manifest.json`` yields 400, not a 500."""
    space = tmp_path / "broken"
    space.mkdir()
    (space / "manifest.json").write_text("{ this is not valid json ")

    inspect = client.get("/api/inspect", params={"space": str(space)})
    assert inspect.status_code == 400, inspect.text

    query = client.post("/api/query", json={"space": str(space), "text": "x"})
    assert query.status_code == 400, query.text
