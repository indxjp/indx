"""T4 — clean install of the wheel + light-core dependency closure (testing-strategy §4.4a)."""

from __future__ import annotations

import pytest
from dockerkit import requires_docker

pytestmark = [pytest.mark.docker, requires_docker]

# Heavy deps that must NEVER appear in the bare-core dependency closure (light-core principle).
BANNED = {"torch", "docling", "qdrant-client", "transformers", "openai", "chromadb"}


def test_cli_runs_in_clean_image(core_image, docker_run) -> None:
    out = docker_run(core_image, ["--version"])
    assert out.returncode == 0
    assert "indx" in out.stdout


def test_core_closure_has_no_heavy_deps(core_image, docker_run) -> None:
    script = (
        "import importlib.metadata as m;"
        "reqs=[r.split(';')[0].split()[0].lower() for r in (m.requires('indx') or []) "
        "if 'extra' not in r];"
        "print('\\n'.join(reqs))"
    )
    # Image ENTRYPOINT is `indx`; override to run the interpreter directly.
    out = docker_run(core_image, ["-c", script], entrypoint="python")
    closure = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    leaked = closure & BANNED
    assert not leaked, f"light core leaked heavy deps: {leaked}"


def test_offline_default_build_in_core_image(core_image, docker_run, tmp_path) -> None:
    # The zero-dep fallback stack runs end to end inside the core image with no extras.
    out = docker_run(
        core_image,
        [
            "build",
            "/data/airgap_smoke",
            "--out",
            "/tmp/space.indx",
            # `--offline` selects the whole zero-dep stack (plaintext/none/hash/jsonl/.indx).
            # The cloud default (docling/openai/qdrant) needs extras, so a core-image build
            # must opt into offline — setting only --parser/--store would still hit the
            # default openai embedder. (There is no `--output` flag; the writer is `--format`.)
            "--offline",
        ],
    )
    assert out.returncode == 0, out.stderr
    assert "→" in out.stdout  # build summary line
