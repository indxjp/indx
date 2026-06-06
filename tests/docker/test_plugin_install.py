"""T4 — third-party plugin discovery via real packaging (§4.4h, file-architecture §6).

Installs the fake plugin distribution into the core image and asserts the registry resolves
`--store fake` through the `indx.stores` entry point — the packaging path the in-process
fake can't exercise.
"""

from __future__ import annotations

import subprocess

import pytest
from dockerkit import REPO_ROOT, requires_docker

pytestmark = [pytest.mark.docker, requires_docker]

PLUGIN_DIR = "/app/plugin"
RESOLVE = (
    "pip install --no-cache-dir /app/plugin >/dev/null && "
    "python -c \"from indx.registry import get_store; print(get_store('fake').name)\""
)


def test_entry_point_store_resolves(core_image) -> None:
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT / 'tests/plugins/fake_plugin'}:{PLUGIN_DIR}:ro",
            "--entrypoint",
            "sh",
            core_image,
            "-c",
            RESOLVE,
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "fake"
