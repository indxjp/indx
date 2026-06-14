"""T12 — the CLI error boundary never leaks a raw vendor traceback.

A missing API key (``openai.AuthenticationError``) or a refused backend connection is NOT an
:class:`~indx.errors.IndxError`, so without the broad fallback in ``_handle_errors`` it would
escape as a stack trace + exit 1. These tests assert the fallback renders a clean
``error: <message>`` line to stderr, exits 1, and emits NO traceback.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import indx.cli.app as app_module
from indx.cli.app import app
from indx.errors import ConfigError, CredentialsError

runner = CliRunner()


class _FakeAuthError(Exception):
    """Stand-in for a vendor SDK exception (e.g. ``openai.AuthenticationError``).

    Deliberately NOT an ``IndxError`` so it exercises the broad ``except Exception`` fallback.
    """


def test_non_indx_vendor_exception_renders_clean_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise _FakeAuthError("Incorrect API key provided")

    # The hidden `build` command body delegates to build_command; make it raise the fake
    # vendor exception the way a missing credential would surface from inside an adapter.
    monkeypatch.setattr(app_module, "build_command", boom)

    src = str(tmp_path)
    result = runner.invoke(app, ["build", src, "--out", f"{src}/out"])

    assert result.exit_code == 1
    # Clean one-line message on stderr, prefixed exactly like every other CLI error.
    assert "error: Incorrect API key provided" in result.stderr
    # No traceback leaked anywhere.
    combined = result.stdout + result.stderr
    assert "Traceback (most recent call last)" not in combined
    assert "_FakeAuthError" not in combined
    # CliRunner only stores the exception when it is re-raised uncaught; the boundary swallows
    # it into a typer.Exit, so nothing should be left dangling as an unhandled error.
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_credentials_error_maps_to_config_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """A typed CredentialsError is a ConfigError → exit 3, with an actionable message."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise CredentialsError("embedder 'openai' could not authenticate: set OPENAI_API_KEY")

    monkeypatch.setattr(app_module, "build_command", boom)

    src = str(tmp_path)
    result = runner.invoke(app, ["build", src, "--out", f"{src}/out"])

    assert result.exit_code == 3  # ConfigError family
    assert "error: embedder 'openai' could not authenticate" in result.stderr
    assert "OPENAI_API_KEY" in result.stderr
    assert "Traceback (most recent call last)" not in (result.stdout + result.stderr)


def test_credentials_error_is_a_config_error() -> None:
    assert issubclass(CredentialsError, ConfigError)


def test_ask_and_query_bad_space_both_exit_2() -> None:
    # Bug #13: a missing SPACE path must be a usage error (exit 2) for BOTH `ask` and `query`.
    # Previously `ask` lacked exists=True, so a bad path slipped past Click and surfaced as an
    # ArchiveError (exit 4), asymmetric with `query`'s exit 2.
    ask_res = runner.invoke(app, ["ask", "q", "/no/such.indx"])
    assert ask_res.exit_code == 2
    query_res = runner.invoke(app, ["query", "q", "/no/such.indx"])
    assert query_res.exit_code == 2
