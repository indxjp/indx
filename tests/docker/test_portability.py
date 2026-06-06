"""T4 — .indx produced in one image opens & queries in another (§4.4f, NFR-PORT-1).

Builds an archive in the 3.12 core image (written to a host volume), then opens it — with
NO network — from an independently-built 3.11 core image. Proves the archive is
self-contained and portable across Python minors / environments, re-loading without
re-processing.
"""

from __future__ import annotations

import subprocess

import pytest
from dockerkit import REPO_ROOT, requires_docker

pytestmark = [pytest.mark.docker, requires_docker]


def test_archive_opens_in_a_different_image(core_image, alt_image, tmp_path) -> None:
    archive = tmp_path / "portable.indx"
    # Build in image A (py3.12), writing into a host-mounted volume.
    build = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/out",
            "--entrypoint",
            "indx",
            core_image,
            "build",
            "/data/airgap_smoke",
            "--out",
            "/out/portable.indx",
            # zero-dep stack; the cloud default needs extras the core image doesn't carry.
            "--offline",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert build.returncode == 0, build.stderr
    assert archive.exists()

    # Open in image B (py3.11), offline. indx is already installed in the image — no pip.
    inspect = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/in",
            "--entrypoint",
            "indx",
            alt_image,
            "inspect",
            "/in/portable.indx",
        ],
        capture_output=True,
        text=True,
    )
    assert inspect.returncode == 0, inspect.stderr
    assert "documents=10" in inspect.stdout
