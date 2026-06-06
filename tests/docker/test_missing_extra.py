"""T4 — selecting an uninstalled slot yields the friendly MissingExtraError (§4.4b)."""

from __future__ import annotations

import pytest
from dockerkit import requires_docker

pytestmark = [pytest.mark.docker, requires_docker]


def test_missing_extra_names_pip_target(core_image, docker_run) -> None:
    out = docker_run(
        core_image,
        # `--offline` makes parser/embedder zero-dep; an explicit `--store qdrant` overrides
        # the preset's jsonl, so the ONLY missing extra is qdrant — otherwise the default
        # docling parser would be the first thing to fail and name the wrong pip target.
        ["build", "/data/airgap_smoke", "--out", "/tmp/o", "--offline", "--store", "qdrant"],
    )
    combined = out.stdout + out.stderr
    assert out.returncode != 0
    assert "pip install indx[qdrant]" in combined
