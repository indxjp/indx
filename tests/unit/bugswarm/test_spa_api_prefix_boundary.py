"""Regression: the SPA catch-all must guard the ``api/`` path boundary, not the substring.

The catch-all in :func:`indx.app.server.create_app` previously used
``full_path.startswith("api")``, which 404'd any client route that merely *began* with the
letters ``api`` (e.g. ``/apiary``, ``/api-docs``, ``/apikeys``) instead of falling back to the
SPA ``index.html``. This test builds the app with a stand-in static bundle so the catch-all is
registered, then asserts api-prefixed-but-not-api paths serve the SPA shell while the genuine
``api/`` boundary is still 404'd by the catch-all.

Runs fully offline: only ``fastapi``/``starlette`` (the app extra) are needed; no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from indx.app import server  # noqa: E402

_SHELL = "<!doctype html><html><body>SPA shell</body></html>"


@pytest.fixture
def bundled_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient over the app with a stand-in SPA bundle so the catch-all is mounted."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(_SHELL)
    monkeypatch.setattr(server, "_static_dir", lambda: static)
    return TestClient(server.create_app())


def test_api_prefixed_client_route_serves_spa_shell(bundled_client: TestClient) -> None:
    # Routes that merely begin with the letters "api" are legitimate client routes and must
    # fall back to the SPA entrypoint, not the 404 JSON body.
    for path in ("/apiary", "/api-docs", "/apikeys"):
        resp = bundled_client.get(path)
        assert resp.status_code == 200, path
        assert "SPA shell" in resp.text, path


def test_real_api_boundary_still_404s_in_catch_all(bundled_client: TestClient) -> None:
    # The genuine ``api/`` boundary (an unknown API path) must never be served the SPA shell.
    resp = bundled_client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert "SPA shell" not in resp.text
