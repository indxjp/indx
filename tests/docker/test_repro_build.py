"""T4 — reproducible build across two fresh containers (§4.4g, NFR-DET-1)."""

from __future__ import annotations

import hashlib

import pytest
from dockerkit import requires_docker

pytestmark = [pytest.mark.docker, requires_docker]

# Build the archive and print its sha256, all inside one container, twice.
HASH_BUILD = (
    # `--offline` = zero-dep stack; the cloud default needs extras the core image lacks.
    "indx build /data/airgap_smoke --out /tmp/s.indx --offline >/dev/null && "
    'python -c "import hashlib,sys;'
    "print(hashlib.sha256(open('/tmp/s.indx','rb').read()).hexdigest())\""
)


def _digest(core_image, docker_run) -> str:
    out = docker_run(core_image, ["-c", HASH_BUILD], entrypoint="sh", network="none")
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_two_clean_builds_are_byte_identical(core_image, docker_run) -> None:
    first = _digest(core_image, docker_run)
    second = _digest(core_image, docker_run)
    assert len(first) == len(hashlib.sha256(b"").hexdigest())
    assert first == second, "deterministic build produced different bytes across containers"
