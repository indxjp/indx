"""Reproducible test/CI sessions (technology-selections §14). Run `nox -l` to list."""

from __future__ import annotations

import nox

nox.options.reuse_existing_virtualenvs = True
PYTHONS = ["3.11", "3.12", "3.13"]


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Fast offline suite: unit + corpus (the default `pytest` run)."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-q", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Ruff lint + format check."""
    session.install("ruff")
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    """mypy strict on the public API."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src/indx")


@nox.session
def integration(session: nox.Session) -> None:
    """T3 — real backends via docker compose (qdrant/postgres/ollama)."""
    session.run(
        "docker", "compose", "-f", "docker/compose.integration.yml", "up",
        "--build", "--abort-on-container-exit", "--exit-code-from", "tests",
        external=True,
    )


@nox.session
def docker(session: nox.Session) -> None:
    """T4 — packaged-wheel distribution tests in clean containers.

    This is also where the `indx demo` corpus wheel-content check belongs: after the wheel
    is built here (hatchling/build are available in this session, unlike the offline dev
    venv), assert that `indx/demo/corpus/*.md|*.txt` is inside `dist/indx-*.whl` — that the
    pyproject force-include/artifacts mapping actually shipped the non-.py data files. The
    offline unit suite (tests/unit/test_demo_packaging.py) can only verify the *source*
    tree + the runtime guard; the real in-wheel assertion needs a built wheel (see
    docs/reference/11-packaging.md "Confirming the corpus actually ships in a wheel").
    """
    session.install("-e", ".[dev]", "build")
    session.run("python", "-m", "build", "--wheel", "--outdir", "dist")
    session.run("pytest", "-m", "docker", "tests/docker", "-q", *session.posargs)


@nox.session
def airgap(session: nox.Session) -> None:
    """Air-gap proof: default stack with no egress, in a network-isolated container."""
    session.run(
        "docker", "compose", "-f", "docker/compose.airgap.yml", "up",
        "--build", "--abort-on-container-exit", "--exit-code-from", "airgap",
        external=True,
    )


@nox.session
def live(session: nox.Session) -> None:
    """T5 — real models end to end. Nightly/manual; may require network."""
    session.install("-e", ".[defaults,dev]")
    session.run("pytest", "-m", "live", *session.posargs)


@nox.session(name="record-fixtures")
def record_fixtures(session: nox.Session) -> None:
    """Regenerate recorded ParsedDoc / embedding fixtures (testing-strategy §3.4). Deliberate."""
    session.install("-e", ".[defaults,dev]")
    session.run("python", "-m", "tests.tools.record_fixtures", *session.posargs)
