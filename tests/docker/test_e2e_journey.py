"""T4/T5 — the packaged user journey build → inspect → query, timed (§4.4e, NFR-PERF-2).

Runs entirely inside the core image (zero-dep stack) so it needs no models/network. The
timing budget is generous for the scaffold's tiny corpus; with the heavy default stack this
test moves to Dockerfile.models and the budget is set from NFR-PERF-2 (<60s on ~10 docs).
"""

from __future__ import annotations

import time

import pytest
from dockerkit import requires_docker

pytestmark = [pytest.mark.docker, requires_docker]

BUILD_BUDGET_S = 60.0  # NFR-PERF-2 activation target

JOURNEY = (
    # `--offline` picks the zero-dep stack (plaintext/none/hash/jsonl); the cloud default
    # (docling/openai/qdrant) needs extras + keys, so the air-gapped journey must opt in.
    "indx build /data/airgap_smoke --out /tmp/s.indx --offline && "
    "indx inspect /tmp/s.indx && "
    "indx query 'sample note about topic' /tmp/s.indx"
)


def test_build_inspect_query_in_image(core_image, docker_run) -> None:
    # Whole journey in ONE container (a fresh `docker run` would lose /tmp/s.indx).
    # network=none proves the default journey needs no egress (US-10 + air-gap).
    start = time.perf_counter()
    out = docker_run(core_image, ["-c", JOURNEY], entrypoint="sh", network="none")
    elapsed = time.perf_counter() - start

    assert out.returncode == 0, out.stderr
    assert "documents=10" in out.stdout
    assert "query:" in out.stdout
    assert elapsed < BUILD_BUDGET_S, f"journey took {elapsed:.1f}s (budget {BUILD_BUDGET_S}s)"
