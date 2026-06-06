"""Air-gap tier — the local / zero-dependency profile runs with no network egress (NFR-PRIV-1, G3).

The zero-config *defaults* are cloud-backed (OpenAI), so this test exercises the explicit
offline profile (plaintext + hash + jsonl, llm/vlm = none) and proves it stays air-gapped.

Two layers of defense:
1. A socket guard installed in-process fails the test if any code opens a network socket
   during the build. This runs in the normal offline suite (marked ``corpus``).
2. In CI this same test is *also* run inside a container with ``network_mode: none``
   (docker/compose.airgap.yml), so the guarantee is kernel-enforced, not just in-process.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from indx import DirectoryPipeline

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"

pytestmark = pytest.mark.corpus


@pytest.fixture
def no_network(monkeypatch):
    """Make any attempt to open a network socket raise (loopback allowed for nothing here)."""
    real_socket = socket.socket

    def guard(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("network egress attempted during an offline-profile build")

    monkeypatch.setattr(socket, "socket", guard)
    yield
    monkeypatch.setattr(socket, "socket", real_socket)


def test_default_build_makes_no_network_calls(no_network, tmp_path) -> None:
    space = DirectoryPipeline(parser="plaintext", llm="none", embedder="hash", store="jsonl").run(
        CORPUS_ROOT / "airgap_smoke"
    )
    assert len(space.documents_) == 10
    assert space.chunks and all(c.embedding is not None for c in space.chunks)
    assert space.manifest.embedding_model == "hash"
