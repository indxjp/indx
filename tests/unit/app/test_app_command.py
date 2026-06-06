"""CLI surface for ``indx app`` (docs/app-spec.md §2, §5).

Covers the command's help, the host/port wiring into ``serve``, the lazy-import guarantee that
``import indx.app`` never pulls ``fastapi``, and the missing-extra contract (a missing
``indx[app]`` raises ``MissingExtraError`` — a plain ``IndxError`` → exit **1**, not 3).

Only the SSE/TestClient-heavy tests need ``fastapi``; the help, lazy-import, and missing-extra
tests run on the core install too, but the whole package self-skips without fastapi per the
canonical extras-absent CI policy.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from indx.cli.app import app

runner = CliRunner()


def test_app_help_shows_command() -> None:
    result = runner.invoke(app, ["app", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--no-open" in result.stdout


def test_app_command_invokes_serve_with_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``indx app --no-open`` calls ``serve`` with the chosen host/port and no browser.

    ``serve`` is patched to a no-op so the test never binds a port or starts uvicorn.
    """
    captured: dict[str, object] = {}

    def fake_serve(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True) -> None:
        captured.update(host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr("indx.app.server.serve", fake_serve)

    result = runner.invoke(app, ["app", "--no-open", "--host", "0.0.0.0", "--port", "9123"])
    assert result.exit_code == 0, result.output
    assert captured == {"host": "0.0.0.0", "port": 9123, "open_browser": False}


def test_import_indx_app_does_not_import_fastapi() -> None:
    """``import indx.app`` must stay safe on a light core install (lazy heavy imports)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, indx.app; assert 'fastapi' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_missing_extra_exits_1_with_clean_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``indx[app]`` extra → MissingExtraError → exit 1 with a clean ``error:``.

    Simulated by making the extra's gate raise, mirroring the require_extra contract
    (MissingExtraError is a plain IndxError mapped to exit 1, NOT the config/registry code 3).
    """
    from indx.errors import MissingExtraError

    def fake_require_extra(slot: str, name: str, extra: str, *modules: str) -> None:
        raise MissingExtraError(slot=slot, name=name, extra=extra)

    # The command calls require_extra imported into the cli.app namespace.
    monkeypatch.setattr("indx.cli.app.require_extra", fake_require_extra)

    result = runner.invoke(app, ["app", "--no-open"])
    assert result.exit_code == 1, result.output
    assert "error:" in result.output
    # A clean typed message, not a raw traceback.
    assert "Traceback" not in result.output
