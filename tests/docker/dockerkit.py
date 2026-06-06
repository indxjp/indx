"""Shared helpers for the Docker tier, importable by test modules (top-level, no package).

pytest's default import mode inserts ``tests/docker`` onto ``sys.path``, so test modules can
``from dockerkit import requires_docker, REPO_ROOT`` without a package/relative import.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_IMAGE = "indx-core:test"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


requires_docker = pytest.mark.skipif(not have("docker"), reason="docker not available")
