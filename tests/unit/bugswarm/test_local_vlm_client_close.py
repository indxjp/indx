"""Regression: LocalVLM must release its ``httpx.Client`` pool deterministically.

The adapter owns a raw ``httpx.Client`` (a live socket pool). It previously exposed no
``close()`` / context-manager, so the pool was only released non-deterministically at GC
(httpx logs an "Unclosed client" warning). These tests pin the fix: an idempotent
``close()`` plus ``__enter__``/``__exit__`` mirroring ChromaStore/PgVectorStore. All run
fully offline with a fake ``httpx`` module.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake ``httpx`` whose client records construction/close calls."""
    state: dict[str, Any] = {"clients": []}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            self.closed = False
            state["clients"].append(self)

        def post(self, url: str, *, json: Any) -> _FakeResponse:
            return _FakeResponse()

        def close(self) -> None:
            self.closed = True

    fake = types.ModuleType("httpx")
    fake.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake)

    import indx.vlm.local as local_mod

    monkeypatch.setattr("indx.utils.lazy.require_extra", lambda *a, **k: None)
    monkeypatch.setattr(local_mod, "require_extra", lambda *a, **k: None)
    return state


def test_close_releases_client_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_httpx(monkeypatch)
    from indx.vlm.local import LocalVLM

    vlm = LocalVLM()
    vlm.describe(b"img")  # opens the pool
    client = state["clients"][-1]
    assert client.closed is False

    vlm.close()
    assert client.closed is True
    assert vlm._client is None

    # Idempotent: a second close() is a no-op and must not raise.
    vlm.close()
    assert vlm._client is None


def test_close_before_use_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch)
    from indx.vlm.local import LocalVLM

    vlm = LocalVLM()
    vlm.close()  # no client ever constructed
    assert vlm._client is None


def test_context_manager_closes_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_httpx(monkeypatch)
    from indx.vlm.local import LocalVLM

    with LocalVLM() as vlm:
        assert vlm.describe(b"img") == "ok"
        client = state["clients"][-1]
        assert client.closed is False

    assert client.closed is True
    assert vlm._client is None


def test_reusable_after_close(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _install_fake_httpx(monkeypatch)
    from indx.vlm.local import LocalVLM

    vlm = LocalVLM()
    vlm.describe(b"img")
    vlm.close()

    # A fresh client is lazily reconstructed on the next call.
    assert vlm.describe(b"img") == "ok"
    assert len(state["clients"]) == 2
    assert state["clients"][0].closed is True
    assert state["clients"][1].closed is False
