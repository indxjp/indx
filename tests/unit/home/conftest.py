"""Shared isolation for the Feature 4 home-DB tests.

Every home test runs against ``$INDX_HOME=<tmp>/home`` so the real ``~/.indx`` is NEVER read or
written. A session-scoped guard records the real home archive path up front and asserts it is
never created during the suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point ``INDX_HOME`` at a per-test tmp dir; assert the real ~/.indx stays untouched."""
    real_home_archive = Path("~/.indx/space.indx").expanduser()
    existed = real_home_archive.exists()
    monkeypatch.setenv("INDX_HOME", str(tmp_path / "home"))
    yield
    # The suite must never create the real home space.
    assert real_home_archive.exists() == existed
