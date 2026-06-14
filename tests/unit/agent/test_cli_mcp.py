"""CLI surface for ``indx mcp`` — the standalone MCP server command.

Covers the help text, the missing-extra contract (a missing ``indx[mcp]`` raises
``MissingExtraError`` → exit 1, like every other extra), and that with the extra present the
command loads the space and runs the server over the chosen transport. The ``mcp`` package is
absent here, so ``connector.serve`` is patched for the success path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from indx.cli.app import app
from indx.core.knowledge_space import KnowledgeSpace

runner = CliRunner()


def test_mcp_help_shows_options() -> None:
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--transport" in result.stdout
    assert "--name" in result.stdout


def test_mcp_missing_extra_exits_1(space: KnowledgeSpace, tmp_path: Path) -> None:
    archive = tmp_path / "kb.indx"
    space.save(str(archive))
    result = runner.invoke(app, ["mcp", str(archive)])
    assert result.exit_code == 1
    assert "mcp" in result.output


def test_mcp_bad_transport_exits_2(space: KnowledgeSpace, tmp_path: Path) -> None:
    # Bug #S11: --transport is constrained to a Choice enum, so a typo is a clean Click usage error
    # (exit 2) at parse time — BEFORE the missing-extra gate (exit 1) ever runs.
    archive = tmp_path / "kb.indx"
    space.save(str(archive))
    result = runner.invoke(app, ["mcp", str(archive), "--transport", "bogus"])
    assert result.exit_code == 2
    assert "streamable-http" in result.output


def test_mcp_serves_when_extra_present(
    space: KnowledgeSpace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "kb.indx"
    space.save(str(archive))

    # Neutralize the extra gate and capture the serve call instead of binding a transport.
    monkeypatch.setattr("indx.cli.app.require_extra", lambda *a, **k: None)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "indx.agent.connector.KnowledgeConnector.serve",
        lambda self, *, transport="stdio", name=None: captured.update(
            transport=transport, server_name=self.name
        ),
    )

    result = runner.invoke(app, ["mcp", str(archive), "--transport", "sse"])
    assert result.exit_code == 0, result.output
    assert captured == {"transport": "sse", "server_name": "kb"}
