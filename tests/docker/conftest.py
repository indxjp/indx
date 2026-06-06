"""Fixtures for the Docker distribution tier (testing-strategy §4).

These tests build the wheel once, build the light-core image, and run the packaged CLI in
clean containers. They SKIP (not fail) when Docker or the build backend is unavailable, so
the marker stays opt-in and a developer without Docker still gets a green default suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from dockerkit import CORE_IMAGE, REPO_ROOT, have, run


@pytest.fixture(scope="session")
def wheel() -> Path:
    """Build the wheel into dist/ once per session."""
    if not have("python"):
        pytest.skip("python not on PATH")
    dist = REPO_ROOT / "dist"
    build = run(["python", "-m", "build", "--wheel", "--outdir", str(dist)], cwd=REPO_ROOT)
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable: {build.stderr[-400:]}")
    wheels = sorted(dist.glob("indx-*.whl"))
    if not wheels:
        pytest.skip("no wheel produced")
    return wheels[-1]


@pytest.fixture(scope="session")
def core_image(wheel: Path) -> str:  # noqa: ARG001 - wheel must exist before building image
    """Build the light-core image (no extras) from the freshly-built wheel."""
    build = run(
        ["docker", "build", "-f", "docker/Dockerfile.core", "-t", CORE_IMAGE, "."],
        cwd=REPO_ROOT,
    )
    if build.returncode != 0:
        pytest.skip(f"core image build failed: {build.stderr[-400:]}")
    return CORE_IMAGE


@pytest.fixture(scope="session")
def alt_image(wheel: Path) -> str:  # noqa: ARG001 - wheel must exist first
    """A second core image on a DIFFERENT Python minor, for cross-env portability (§4.4f)."""
    tag = "indx-core-py311:test"
    build = run(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile.core",
            "--build-arg",
            "PYTHON_VERSION=3.11",
            "-t",
            tag,
            ".",
        ],
        cwd=REPO_ROOT,
    )
    if build.returncode != 0:
        pytest.skip(f"alt image build failed: {build.stderr[-400:]}")
    return tag


@pytest.fixture
def docker_run():
    def _go(
        image: str,
        args: list[str],
        *,
        network: str | None = None,
        entrypoint: str | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = ["docker", "run", "--rm"]
        if network is not None:
            cmd += ["--network", network]
        if entrypoint is not None:
            cmd += ["--entrypoint", entrypoint]
        cmd += [image, *args]
        return run(cmd)

    return _go
