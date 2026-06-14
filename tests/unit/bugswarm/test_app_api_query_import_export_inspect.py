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

from indx.app.models import BuildRequest  # noqa: E402
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


def test_browse_rejects_path_outside_workspace(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /api/browse`` refuses a path outside the working-directory tree (no FS enumeration).

    Before the fix ``?path=/etc`` returned 200 enumerating the directory; the picker now applies
    the same containment guard as ``/api/export`` and rejects with 400.
    """
    monkeypatch.chdir(tmp_path)
    resp = client.get("/api/browse", params={"path": "/etc"})
    assert resp.status_code == 400, resp.text
    # The working-directory tree itself stays browsable.
    (tmp_path / "sub").mkdir()
    ok = client.get("/api/browse", params={"path": str(tmp_path / "sub")})
    assert ok.status_code == 200, ok.text


def test_query_strips_embedding_vector(client: TestClient, corpus: Path, tmp_path: Path) -> None:
    """``POST /api/query`` must not inline the dense embedding vector on each hit/neighbor.

    Before the fix every hit's ``chunk.embedding`` (and each neighbor's) serialized as a full
    ``list[float]`` of the embedder width — dead weight the UI never uses. It is now ``None``.
    """
    out = tmp_path / "ks"
    _build_space(client, corpus, out)

    resp = client.post("/api/query", json={"space": str(out), "text": "handbook", "k": 3})
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]
    assert hits, "expected at least one hit for the offline lexical retriever"
    for hit in hits:
        assert hit["chunk"].get("embedding") is None, hit["chunk"]
        for neighbor in hit.get("neighbors", []):
            assert neighbor.get("embedding") is None, neighbor


def test_build_sse_terminal_frame_survives_disconnect_falsepositive(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    """A spurious ``is_disconnected`` true right after ``done`` must not drop the terminal frame.

    Drives the build endpoint's inner ``stream`` generator directly with a fake request whose
    ``is_disconnected`` flips true on the probe that follows the ``done`` frame — the exact latch
    that previously swallowed the terminal frame. The consumer now yields a dequeued frame before
    probing, so ``done`` is still delivered. Before the reorder this collection lacks ``done``.
    """
    import asyncio

    from indx.app.api import build_router

    router = build_router()
    build_route = next(r for r in router.routes if getattr(r, "path", None) == "/build")

    out = tmp_path / "ks"
    req = BuildRequest(directory=str(corpus), out=str(out), offline=True)

    delivered: list[str] = []
    # The offline build emits a deterministic frame sequence: ``start`` + six ``stage`` frames +
    # the terminal ``done`` = eight frames, then the ``None`` sentinel. The consumer calls
    # ``is_disconnected`` exactly once per dequeued frame; we latch it true on the eighth call —
    # the one paired with the ``done`` frame. On the broken (check-before-yield) order that probe
    # runs *before* ``done`` is yielded and drops it; on the fixed (yield-before-check) order the
    # probe runs *after* ``done`` is already delivered, so it survives.
    terminal_call = {"n": 8}
    probe_calls = {"n": 0}

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            probe_calls["n"] += 1
            return probe_calls["n"] >= terminal_call["n"]

    async def _drive() -> None:
        # ``build`` returns a StreamingResponse; ``body_iterator`` is the inner ``stream`` gen.
        response = build_route.endpoint(req, _FakeRequest())  # type: ignore[arg-type]
        async for frame in response.body_iterator:
            line = frame.split("\n", 1)[0]
            if line.startswith("event:"):
                delivered.append(line.split(":", 1)[1].strip())

    asyncio.run(_drive())

    assert "done" in delivered, delivered
    assert "error" not in delivered, delivered
