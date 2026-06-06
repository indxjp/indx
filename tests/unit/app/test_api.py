"""End-to-end coverage of the ``indx app`` HTTP API (docs/app-spec.md §3).

Drives the real FastAPI app via ``starlette.testclient.TestClient`` over an offline build of a
tmp corpus — no network, no extras, mirroring the zero-dep core stack the CLI tests use. Each
endpoint is checked against the documented response shapes (which embed the real core models:
``SpaceStats``, ``SearchHit``, ``Manifest``). The whole module self-skips when ``fastapi`` is
absent (see ``conftest.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from indx import __version__

# --------------------------------------------------------------------------- helpers


def _sse_events(client: TestClient, body: dict[str, object]) -> list[tuple[str, dict]]:
    """POST /api/build and parse the SSE stream into ``(event, data)`` pairs."""
    events: list[tuple[str, dict]] = []
    current = ""
    with client.stream("POST", "/api/build", json=body) as resp:
        assert resp.status_code == 200, resp.read()
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((current, json.loads(line.split(":", 1)[1].strip())))
    return events


def _build_space(client: TestClient, corpus: Path, out: Path) -> dict:
    """Run an offline SSE build and return the ``done`` summary payload."""
    events = _sse_events(client, {"directory": str(corpus), "out": str(out), "offline": True})
    done = [data for event, data in events if event == "done"]
    assert done, f"no done event in stream: {[e for e, _ in events]}"
    return done[0]


# --------------------------------------------------------------------------- health


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert isinstance(body["static"], bool)


# --------------------------------------------------------------------------- components


def test_components_every_slot_present_and_core_installed(client: TestClient) -> None:
    body = client.get("/api/components").json()
    for slot in ("parser", "llm", "vlm", "embedder", "store", "output"):
        assert isinstance(body[slot], list) and body[slot], slot
        for comp in body[slot]:
            assert set(comp) == {"name", "builtin", "extra", "installed"}
            # Every zero-dep core backend (no extra) is always reported installed.
            if comp["extra"] is None:
                assert comp["installed"] is True, (slot, comp["name"])
    # Documented presets, with the writer slot exposed as ``output``.
    assert body["defaults"]["parser"] == "docling"
    assert body["defaults"]["output"] == ".indx"
    assert body["offline"] == {
        "parser": "plaintext",
        "llm": "none",
        "vlm": "none",
        "embedder": "hash",
        "store": "jsonl",
        "output": ".indx",
    }


def test_components_extra_gated_backend_reflects_find_spec(client: TestClient) -> None:
    """An extra-gated backend carries its pip-extra name and a find_spec-driven flag."""
    import importlib.util

    stores = {c["name"]: c for c in client.get("/api/components").json()["store"]}
    qdrant = stores["qdrant"]
    assert qdrant["extra"] == "qdrant"
    # ``installed`` mirrors whether the adapter's vendor module imports (qdrant_client).
    expected = importlib.util.find_spec("qdrant_client") is not None
    assert qdrant["installed"] is expected
    # The zero-dep core store is always installed and carries no extra.
    assert stores["jsonl"]["extra"] is None
    assert stores["jsonl"]["installed"] is True


# --------------------------------------------------------------------------- config


def test_config_get_returns_defaults(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert "config" in body and "path" in body
    cfg = body["config"]
    # The product defaults (cloud stack) are returned when no indx.toml is present.
    assert cfg["parser"]["engine"] == "docling"
    assert cfg["store"]["backend"] == "qdrant"


def test_config_validate_valid_and_invalid(client: TestClient) -> None:
    ok = client.post("/api/config/validate", json={"parser": {"engine": "plaintext"}}).json()
    assert ok == {"valid": True, "errors": []}

    bad = client.post("/api/config/validate", json={"parser": "not-a-dict"}).json()
    assert bad["valid"] is False
    assert bad["errors"] and any("parser" in e for e in bad["errors"])


def test_config_put_round_trip(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PUT writes indx.toml under the server CWD; a subsequent GET reads it back."""
    monkeypatch.chdir(tmp_path)
    payload = {
        "config": {
            "parser": {"engine": "plaintext"},
            "enrich": {"llm": "none", "vlm": "none"},
            "embed": {"model": "hash"},
            "store": {"backend": "jsonl"},
            "output": {"format": ".indx"},
        }
    }
    put = client.put("/api/config", json=payload)
    assert put.status_code == 200, put.text
    written = Path(put.json()["path"])
    assert written.is_file()
    assert written.parent == tmp_path.resolve()

    # GET the freshly written file: the offline stack round-trips through load_config.
    got = client.get("/api/config", params={"path": str(written)}).json()
    assert got["config"]["parser"]["engine"] == "plaintext"
    assert got["config"]["store"]["backend"] == "jsonl"


def test_config_put_rejects_path_traversal(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resp = client.put(
        "/api/config",
        json={"path": "../escape.toml", "config": {"parser": {"engine": "plaintext"}}},
    )
    assert resp.status_code == 400


def test_config_put_round_trips_control_chars(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passthrough value with newlines/tabs/quotes survives the TOML write→read round-trip.

    The hand-rolled TOML writer must escape control chars; an unescaped newline would make the
    written indx.toml fail to re-parse (regression guard for the _fmt_str escaping fix).
    """
    monkeypatch.chdir(tmp_path)
    nasty = 'line1\nline2\ttab "quoted" \\slash'
    payload = {
        "config": {
            "parser": {"engine": "plaintext"},
            "store": {"backend": "qdrant", "qdrant": {"note": nasty}},
        }
    }
    put = client.put("/api/config", json=payload)
    assert put.status_code == 200, put.text
    got = client.get("/api/config", params={"path": put.json()["path"]}).json()
    assert got["config"]["store"]["qdrant"]["note"] == nasty


def test_config_get_honors_cli_config_env(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no explicit ?path=, GET /config defaults to the file `indx app --config` named.

    The CLI stashes its --config in INDX_APP_CONFIG; the editor must open against that file,
    not ./indx.toml (regression guard for the previously-dead --config flag).
    """
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[store]\nbackend = "qdrant"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # ./indx.toml does NOT exist here
    monkeypatch.setenv("INDX_APP_CONFIG", str(cfg))
    got = client.get("/api/config").json()
    assert got["path"] == str(cfg)
    assert got["config"]["store"]["backend"] == "qdrant"


# --------------------------------------------------------------------------- build (SSE)


def test_build_sse_offline_streams_stages_and_done(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ks"
    events = _sse_events(client, {"directory": str(corpus), "out": str(out), "offline": True})
    names = [event for event, _ in events]
    assert names[0] == "start"
    assert "done" in names and "error" not in names

    start = events[0][1]
    assert start["components"]["parser"] == "plaintext"
    assert start["out"] == str(out)

    stages_streamed = [d["name"] for e, d in events if e == "stage"]
    assert stages_streamed == ["walk", "parse", "chunk", "relate", "enrich", "embed-pack"]

    done = [d for e, d in events if e == "done"][0]
    assert done["counts"]["docs"] >= 2
    assert done["counts"]["chunks"] > 0
    # The done summary's stage list includes a final synthetic ``write`` entry.
    summary_stages = [s["name"] for s in done["stages"]]
    assert summary_stages[-1] == "write"
    assert "parse" in summary_stages
    assert done["components"]["store"] == "jsonl"


def test_build_sse_dry_run_emits_single_plan_event(client: TestClient, corpus: Path) -> None:
    """``dry_run: true`` is terminal with a single ``plan`` event (no models run)."""
    events = _sse_events(client, {"directory": str(corpus), "offline": True, "dry_run": True})
    assert [e for e, _ in events] == ["plan"]
    plan = events[0][1]
    assert plan["root"]
    assert len(plan["documents"]) >= 2
    assert {"id", "path", "type", "folder", "size_bytes"} <= set(plan["documents"][0])
    assert isinstance(plan["embed"], bool) and isinstance(plan["enrich"], bool)


# --------------------------------------------------------------------------- dry-run


def test_dry_run_endpoint(client: TestClient, corpus: Path) -> None:
    resp = client.post("/api/dry-run", json={"directory": str(corpus), "offline": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["documents"]) >= 2
    assert body["components"]["parser"] == "plaintext"
    assert isinstance(body["folders"], list)


# --------------------------------------------------------------------------- inspect


def test_inspect_built_space(client: TestClient, corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ks"
    done = _build_space(client, corpus, out)

    body = client.get("/api/inspect", params={"space": str(out)}).json()
    # stats is exactly SpaceStats; documents count matches the build summary.
    assert body["stats"]["documents"] == done["counts"]["docs"]
    assert set(body["stats"]) == {
        "documents",
        "chunks",
        "relations",
        "embeddings",
        "embed_dim",
        "types",
        "bytes_source",
    }
    assert len(body["documents"]) == done["counts"]["docs"]
    assert {"id", "type", "path", "folder", "topics", "tags", "chunks"} <= set(body["documents"][0])
    assert isinstance(body["types"], dict) and isinstance(body["relations"], dict)
    assert "manifest" in body


# --------------------------------------------------------------------------- query


def test_query_returns_hits(client: TestClient, corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ks"
    _build_space(client, corpus, out)

    resp = client.post(
        "/api/query",
        json={"space": str(out), "text": "onboarding handbook", "k": 2},
    )
    assert resp.status_code == 200, resp.text
    hits = resp.json()["hits"]
    assert hits, "expected at least one hit for the offline lexical retriever"
    hit = hits[0]
    # Each hit serializes the real SearchHit surface (chunk/score/neighbors).
    assert {"chunk", "score", "neighbors"} <= set(hit)
    assert isinstance(hit["score"], float)
    assert "text" in hit["chunk"]


# --------------------------------------------------------------------------- demo


def test_demo_builds_offline_and_returns_summary(client: TestClient) -> None:
    resp = client.post("/api/demo")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    out = Path(body["out"])
    assert out.is_dir()
    summary = body["summary"]
    assert summary["counts"]["docs"] > 0
    assert summary["components"]["parser"] == "plaintext"
    # The returned space is immediately inspectable.
    inspected = client.get("/api/inspect", params={"space": str(out)}).json()
    assert inspected["stats"]["documents"] == summary["counts"]["docs"]


# --------------------------------------------------------------------------- browse


def test_browse_lists_directories_first(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "file.txt").write_text("x")
    body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    assert body["path"] == str(tmp_path.resolve())
    assert body["parent"] == str(tmp_path.resolve().parent)
    names = [e["name"] for e in body["entries"]]
    assert {"a_dir", "z_dir", "file.txt"} <= set(names)
    # Directories sort before files.
    dir_flags = [e["is_dir"] for e in body["entries"]]
    assert dir_flags == sorted(dir_flags, reverse=True)


def test_browse_defaults_to_cwd(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    body = client.get("/api/browse").json()
    assert body["path"] == str(tmp_path.resolve())
    assert "sub" in [e["name"] for e in body["entries"]]
