"""Shared fixtures for the ``indx app`` API tests (docs/app-spec.md §3).

Every module under ``tests/unit/app`` self-skips when ``fastapi`` is absent, so the canonical
extras-absent CI stays green (docs/app-spec.md §7). ``starlette.testclient.TestClient`` rides on
the already-present ``httpx``, so no extra runtime dependency is pulled in for the test harness.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

# Self-skip the whole package at collection time when the app extra is not installed.
pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from indx.app.server import create_app  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient over the real FastAPI app (``/api`` router mounted)."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny two-file corpus the offline core stack can build with zero installs."""
    src = tmp_path / "src"
    (src / "notes").mkdir(parents=True)
    (src / "notes" / "alpha.md").write_text("# Alpha\n\nThe onboarding handbook lives here.\n")
    (src / "beta.md").write_text("# Beta\n\nSecurity policy details.\n")
    return src
