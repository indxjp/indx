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


def test_build_writes_exactly_one_archive_named_for_request(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    """The app build path serializes the archive once, named ``f"{req.name}.indx"`` (issue #22).

    The build previously wrote twice — ``run()`` sealed a default ``handbook.indx`` *and* the
    explicit writer wrote ``<name>.indx`` — breaking ``_resolve_export_path``'s single-archive
    invariant. Assert one ``*.indx`` carrying the requested name.
    """
    out = tmp_path / "ks"
    events = _sse_events(
        client,
        {"directory": str(corpus), "out": str(out), "offline": True, "name": "corpus"},
    )
    assert "done" in [e for e, _ in events] and "error" not in [e for e, _ in events]
    archives = sorted(p.name for p in out.glob("*.indx"))
    assert archives == ["corpus.indx"]


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


def test_dry_run_missing_directory_returns_400(client: TestClient, tmp_path: Path) -> None:
    """A non-directory path yields a clean 400, not a 500 (issue #24).

    ``plan()`` raises a bare stdlib ``NotADirectoryError``/``FileNotFoundError`` for a path that
    isn't a dir/zip; the endpoint must map it to a 4xx like /build and /inspect already do.
    """
    missing = tmp_path / "nope-dir"
    resp = client.post("/api/dry-run", json={"directory": str(missing), "offline": True})
    assert resp.status_code == 400, resp.text


def test_config_put_rejects_unknown_section(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong-shaped body (unknown top-level section) is rejected, not silently defaulted.

    Regression for issue #24: ``Config`` is ``extra="allow"`` for ``[store.<backend>]`` sub-tables,
    so a body like ``{"preset": "local"}`` would otherwise validate, drop ``preset``, and persist a
    full *default* config over the user's saved stack. Assert a 422 and that no file was written.
    """
    monkeypatch.chdir(tmp_path)
    resp = client.put("/api/config", json={"config": {"preset": "local"}})
    assert resp.status_code == 422, resp.text
    assert "preset" in str(resp.json()["detail"])
    assert not (tmp_path / "indx.toml").exists()  # no clobber


def test_inject_space_collection_derives_per_space_name(tmp_path: Path) -> None:
    """Each Qdrant build gets a collection named for its output dir, unless one is pinned."""
    from indx.app.api import _inject_space_collection
    from indx.config import load_config

    out = tmp_path / "indx-app-6mh0erv8"
    cfg = load_config(None, overrides={"store": "qdrant"})
    _inject_space_collection(cfg, out)
    assert cfg.store.options().get("collection") == "indx-app-6mh0erv8"

    # An explicit pin wins; injection is a no-op.
    pinned = load_config(None, overrides={"store": "qdrant"})
    pinned.store.qdrant = {"collection": "mine"}
    _inject_space_collection(pinned, out)
    assert pinned.store.options().get("collection") == "mine"

    # Non-qdrant backends are untouched.
    jsonl = load_config(None, overrides={"store": "jsonl"})
    _inject_space_collection(jsonl, out)
    assert "collection" not in jsonl.store.options()


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
        "unindexed_documents",
        "unindexed_paths",
    }
    assert len(body["documents"]) == done["counts"]["docs"]
    assert {"id", "type", "path", "folder", "topics", "tags", "chunks"} <= set(body["documents"][0])
    assert isinstance(body["types"], dict) and isinstance(body["relations"], dict)
    assert "manifest" in body

    # edges are net-new: each is a document-to-document relation that joins with `documents`.
    edges = body["edges"]
    assert isinstance(edges, list)
    assert len(edges) == done["counts"]["relations"]
    # edge_total reports the true relation count (edges may be capped) so the UI can say "N of M".
    assert body["edge_total"] == done["counts"]["relations"]
    assert body["edge_total"] >= len(edges)
    doc_ids = {d["id"] for d in body["documents"]}
    for edge in edges:
        assert set(edge) == {"source_id", "target_id", "type"}
        # Each endpoint joins back to a real document, so the graph renders connected.
        assert edge["source_id"] in doc_ids
        assert edge["target_id"] in doc_ids
        assert edge["type"]
    # The histogram (kept for the legend) still agrees with the edge list per type.
    if edges:
        from collections import Counter

        by_type = Counter(e["type"] for e in edges)
        assert dict(by_type) == body["relations"]


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


def test_browse_lists_directories_first(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Browse under the server CWD: the containment guard only allows the working-directory tree
    # (or an app-owned indx-app-* temp dir), so serve from tmp_path before browsing it.
    monkeypatch.chdir(tmp_path)
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


# --------------------------------------------------------------------------- agent


def test_agent_frameworks_reports_install_state(client: TestClient) -> None:
    """``/agent/frameworks`` lists every adapter with a find_spec-driven installed flag."""
    import importlib.util

    body = client.get("/api/agent/frameworks").json()
    by_name = {f["name"]: f for f in body}
    assert {"langchain", "openai-agents", "pydantic-ai", "claude-agent", "mcp"} <= set(by_name)
    for info in body:
        assert set(info) == {"name", "extra", "installed"}
        assert isinstance(info["installed"], bool)
    # langchain badge mirrors whether langchain_core imports.
    expected_lc = importlib.util.find_spec("langchain_core") is not None
    assert by_name["langchain"]["installed"] is expected_lc
    assert by_name["langchain"]["extra"] == "langchain"
    # mcp is installed iff EITHER fastmcp OR mcp imports.
    expected_mcp = (
        importlib.util.find_spec("fastmcp") is not None
        or importlib.util.find_spec("mcp") is not None
    )
    assert by_name["mcp"]["installed"] is expected_mcp


def test_agent_tools_returns_static_tool_defs(client: TestClient) -> None:
    body = client.get("/api/agent/tools").json()
    names = {t["name"] for t in body}
    assert names == {"indx_search", "indx_overview", "indx_get_document"}
    for tool in body:
        assert {"name", "description", "parameters"} <= set(tool)
        assert tool["parameters"]["type"] == "object"


def test_agent_snippets_are_copy_paste_strings(client: TestClient) -> None:
    body = client.get("/api/agent/snippets").json()
    assert set(body) == {"python", "cli", "langchain", "llamaindex", "mcp"}
    assert "from indx.agent import connect" in body["python"]
    assert "indx mcp" in body["cli"]


def test_agent_overview_over_offline_space(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ks"
    done = _build_space(client, corpus, out)

    resp = client.post("/api/agent/overview", json={"space": str(out), "sample": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The real SpaceOverview shape (connector.overview).
    assert body["documents"] == done["counts"]["docs"]
    assert body["chunks"] == done["counts"]["chunks"]
    assert isinstance(body["types"], dict)
    assert len(body["sample_documents"]) <= 5
    assert {"id", "path"} <= set(body["sample_documents"][0])


def test_agent_search_returns_flat_hits(client: TestClient, corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ks"
    _build_space(client, corpus, out)

    resp = client.post(
        "/api/agent/search",
        json={"space": str(out), "text": "onboarding handbook", "k": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "onboarding handbook"
    assert body["count"] == len(body["hits"])
    assert body["hits"], "expected at least one hit from the offline lexical retriever"
    hit = body["hits"][0]
    # The flat Hit shape: source is a plain path string (distinct from /query's nested chunk).
    assert {"chunk_id", "document_id", "score", "text"} <= set(hit)
    assert isinstance(hit["score"], float)
    assert hit["source"] is None or isinstance(hit["source"], str)


def test_agent_document_returns_detail_and_error_sentinel(
    client: TestClient, corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "ks"
    _build_space(client, corpus, out)

    # Resolve a known document via its path suffix, like the connector does.
    inspected = client.get("/api/inspect", params={"space": str(out)}).json()
    a_path = inspected["documents"][0]["path"]

    ok = client.post("/api/agent/document", json={"space": str(out), "path_or_id": a_path})
    assert ok.status_code == 200, ok.text
    detail = ok.json()
    assert detail["path"] == a_path
    assert {"chunk_count", "text"} <= set(detail)
    assert "error" not in detail

    # A missing id returns the {error} sentinel at HTTP 200, never a 500.
    missing = client.post(
        "/api/agent/document", json={"space": str(out), "path_or_id": "does/not/exist.md"}
    )
    assert missing.status_code == 200, missing.text
    sentinel = missing.json()
    assert "error" in sentinel
    assert "does/not/exist.md" in sentinel["error"]


# --------------------------------------------------------------------------- export


def test_export_streams_built_space_as_download(
    client: TestClient, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/export streams the built ``.indx`` as an attachment with a non-empty body."""
    out = tmp_path / "ks"
    _build_space(client, corpus, out)
    # The export guard resolves ``space`` under the server CWD, so build/serve from tmp_path.
    monkeypatch.chdir(tmp_path)

    resp = client.get("/api/export", params={"space": str(out)})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/octet-stream"
    disp = resp.headers["content-disposition"]
    assert "attachment" in disp
    assert ".indx" in disp
    # A real self-contained ``.indx`` is a zip (PK magic) with a non-empty body.
    body = resp.content
    assert body, "export body should be non-empty"
    assert body[:2] == b"PK"


def test_export_rejects_path_traversal(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``space`` that escapes the server CWD tree is rejected with 400, not streamed."""
    monkeypatch.chdir(tmp_path)
    resp = client.get("/api/export", params={"space": "../../../../etc/passwd"})
    assert resp.status_code == 400


def test_export_streams_app_owned_temp_space(
    client: TestClient, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export reaches an app-owned ``indx-app-*`` build dir even though it's outside the CWD.

    Every in-app build/demo/import writes to ``tempfile.mkdtemp(prefix="indx-app-...")`` under the
    system temp root, so the CWD-only guard made export unreachable for anything the app produced.
    """
    import shutil
    import tempfile

    # Serve from an unrelated CWD so the build dir is reachable *only* via the temp-root branch.
    monkeypatch.chdir(tmp_path)
    app_dir = Path(tempfile.mkdtemp(prefix="indx-app-"))
    try:
        _build_space(client, corpus, app_dir)
        resp = client.get("/api/export", params={"space": str(app_dir)})
        assert resp.status_code == 200, resp.text
        assert resp.content[:2] == b"PK"
    finally:
        shutil.rmtree(app_dir, ignore_errors=True)


def test_export_rejects_non_app_temp_dir(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temp-root allowance is scoped to ``indx-app-*`` dirs — an arbitrary /tmp path is 400.

    This keeps the export from streaming any file that merely happens to live under the system
    temp root (only the app's own build dirs are exportable).
    """
    import shutil
    import tempfile

    monkeypatch.chdir(tmp_path)
    other = Path(tempfile.mkdtemp(prefix="notapp-"))
    try:
        resp = client.get("/api/export", params={"space": str(other)})
        assert resp.status_code == 400
    finally:
        shutil.rmtree(other, ignore_errors=True)


# --------------------------------------------------------------------------- import


def test_import_saves_upload_under_workdir(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/import saves a small upload under the app-owned work dir and reports kind."""
    pytest.importorskip("multipart")  # python-multipart backs fastapi's multipart parsing
    from indx.app import api

    workdir = tmp_path / "import-work"
    monkeypatch.setenv("INDX_APP_IMPORT_DIR", str(workdir))
    # ``_import_workdir`` is lru_cache'd for the server lifetime; drop any cached value so this
    # test's env override is honored regardless of test ordering.
    api._import_workdir.cache_clear()

    # A real ``.indx`` upload (a zip carrying manifest.json) is classified as a ready-to-inspect
    # space. Import validates the archive structure, so fake bytes would be rejected (see below).
    indx_bytes = _minimal_indx_archive()
    resp = client.post(
        "/api/import",
        files={"file": ("my space.indx", indx_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    saved = Path(body["path"])
    assert body["kind"] == "space"
    assert saved.is_file()
    assert saved.read_bytes() == indx_bytes
    # The saved file lives under the app-owned work dir, with a sanitized (no-space) basename.
    assert saved.parent == workdir.resolve()
    assert " " not in saved.name

    # A non-``.indx`` upload is classified as raw build input.
    raw = client.post(
        "/api/import",
        files={"file": ("notes.zip", b"raw-bytes", "application/zip")},
    )
    assert raw.status_code == 200, raw.text
    assert raw.json()["kind"] == "raw"


def _minimal_indx_archive() -> bytes:
    """Smallest bytes that pass the import validator: a zip carrying ``manifest.json``."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
    return buf.getvalue()


def test_import_rejects_bogus_indx_archive(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.indx`` upload that is not a real archive is rejected up front with a clean 400.

    Regression guard for the dogfood finding where a truncated/mis-renamed ``.indx`` was reported
    as a ``space``; the UI opened it optimistically and the failure only surfaced one step later as
    a raw loader error referencing the internal temp path. Reject at import instead.
    """
    pytest.importorskip("multipart")
    from indx.app import api

    workdir = tmp_path / "import-work"
    monkeypatch.setenv("INDX_APP_IMPORT_DIR", str(workdir))
    api._import_workdir.cache_clear()

    # Not a zip at all.
    bogus = client.post(
        "/api/import",
        files={"file": ("broken.indx", b"this is not a zip", "application/octet-stream")},
    )
    assert bogus.status_code == 400, bogus.text
    assert "not a valid .indx archive" in bogus.json()["detail"]

    # A valid zip, but missing manifest.json (e.g. an unrelated archive renamed to .indx).
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.txt", "hello")
    no_manifest = client.post(
        "/api/import",
        files={"file": ("renamed.indx", buf.getvalue(), "application/octet-stream")},
    )
    assert no_manifest.status_code == 400, no_manifest.text

    # The rejected uploads do not linger in the work dir.
    if workdir.exists():
        assert not any(p.suffix == ".indx" for p in workdir.iterdir())


def test_build_worker_cancels_and_removes_orphan_out_dir(corpus: Path) -> None:
    """A cooperative stop bails the build at the next stage and deletes its temp output dir.

    Models bug 2: on client disconnect the worker must not run to completion, and the temp output
    dir it created (but never handed to the client) must not be orphaned on disk.
    """
    import threading

    from indx.app import api
    from indx.app.models import BuildRequest

    stop = threading.Event()
    frames: list[tuple[str | None, dict | None]] = []
    out_dir_holder: dict[str, str] = {}

    def emit(event: str | None, data: dict | None) -> None:
        frames.append((event, data))
        if event == "start":
            out_dir_holder["out"] = data["out"]  # type: ignore[index]
        if event == "stage":
            # Simulate the client disconnecting after the first stage starts.
            stop.set()

    req = BuildRequest(directory=str(corpus), offline=True)  # no ``out`` -> mkdtemp temp dir
    api._run_build_worker(req, emit, stop)

    events = [e for e, _ in frames]
    assert "done" not in events, events  # cancelled, never completed
    assert events[-1] is None  # the terminal sentinel still fires
    # The orphan temp output dir the worker created was cleaned up on cancel.
    out_dir = out_dir_holder.get("out")
    assert out_dir is not None
    assert not Path(out_dir).exists(), out_dir


def test_import_workdir_default_is_private_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, the import dir is an unpredictable mkdtemp dir at mode 0700.

    A predictable, world-traversable shared dir would let a co-tenant read other users' imported
    ``.indx`` spaces (or pre-plant a symlink); mkdtemp gives a fresh random 0700 dir each process.
    """
    import os
    import stat

    from indx.app import api

    monkeypatch.delenv("INDX_APP_IMPORT_DIR", raising=False)
    api._import_workdir.cache_clear()
    try:
        workdir = api._import_workdir()
        assert workdir.is_dir()
        assert workdir.name.startswith(api._IMPORT_WORKDIR_PREFIX)
        mode = stat.S_IMODE(os.stat(workdir).st_mode)
        assert mode == 0o700, oct(mode)
        # lru_cache keeps the dir stable across requests.
        assert api._import_workdir() == workdir
    finally:
        api._import_workdir.cache_clear()


def test_import_workdir_refuses_symlinked_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing symlink at the override path is refused (no redirect of uploads)."""
    from indx.app import api

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "planted"
    link.symlink_to(real, target_is_directory=True)

    monkeypatch.setenv("INDX_APP_IMPORT_DIR", str(link))
    api._import_workdir.cache_clear()
    try:
        with pytest.raises(ValueError, match="symlink"):
            api._import_workdir()
    finally:
        api._import_workdir.cache_clear()


def test_import_cleans_up_partial_file_on_write_error(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write failure mid-upload removes the partial file instead of orphaning it on disk.

    Cleanup must cover any abort (OSError/ClientDisconnect), not only the over-cap 413 case.
    """
    pytest.importorskip("multipart")
    from indx.app import api

    workdir = tmp_path / "import-work"
    monkeypatch.setenv("INDX_APP_IMPORT_DIR", str(workdir))
    api._import_workdir.cache_clear()

    real_open = Path.open

    class _BoomHandle:
        """Wraps a real file handle but fails every ``write`` (simulating ENOSPC mid-stream)."""

        def __init__(self, fh: object) -> None:
            self._fh = fh

        def write(self, _chunk: bytes) -> int:
            raise OSError("simulated ENOSPC")

        def close(self) -> None:
            self._fh.close()

    class _BoomCtx:
        def __init__(self, path: Path, args: tuple, kwargs: dict) -> None:
            self._real = real_open(path, *args, **kwargs)

        def __enter__(self) -> _BoomHandle:
            return _BoomHandle(self._real.__enter__())

        def __exit__(self, *exc: object) -> object:
            return self._real.__exit__(*exc)

    def boom_open(self: Path, *args: object, **kwargs: object) -> object:
        return _BoomCtx(self, args, kwargs)

    monkeypatch.setattr(Path, "open", boom_open)

    with pytest.raises(OSError, match="simulated ENOSPC"):
        client.post(
            "/api/import",
            files={"file": ("notes.zip", b"raw-bytes", "application/zip")},
        )

    monkeypatch.undo()
    api._import_workdir.cache_clear()
    # No partial file was left behind under the work dir.
    assert list(workdir.iterdir()) == []
