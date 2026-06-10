"""Regression tests for registry._resolve diagnostics (bug-swarm).

Two confirmed defects in :mod:`indx.registry.registry`:

1. The RegistryError for an unknown name listed only builtin names, omitting installed
   third-party plugins advertised under the slot's entry-point group(s).
2. A broad ``except ModuleNotFoundError`` converted *any* import failure into a
   missing-extra hint, masking a typo'd / refactored BUILTINS module path (a code bug no
   ``pip install`` can fix) behind an install instruction.

Both run fully offline: plugin discovery and the builtin table are monkeypatched.
"""

from __future__ import annotations

import pytest

from indx.errors import MissingExtraError, RegistryError
from indx.registry import registry


def test_unknown_name_lists_installed_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate an installed third-party store plugin named "fake" without touching the
    # real entry-point metadata.
    monkeypatch.setattr(
        registry,
        "discover",
        lambda group: {"fake": "fake_plugin:FakeStore"} if group == "indx.stores" else {},
    )
    monkeypatch.setattr(registry, "load_plugin", lambda group, name: None)

    with pytest.raises(RegistryError) as exc:
        registry._resolve("store", "fak")

    msg = str(exc.value)
    assert "unknown store 'fak'" in msg
    assert "fake" in msg  # the installed plugin is advertised in the diagnostic
    assert "jsonl" in msg  # builtins still listed too


def test_typod_builtin_path_propagates_not_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A typo'd adapter module path makes the adapter module itself unimportable. That is a
    # packaging/refactor bug, not an absent extra, so the raw ModuleNotFoundError must
    # propagate rather than being rewritten as a misleading "pip install indx[...]" hint.
    monkeypatch.setitem(registry.BUILTINS["store"], "qdrant", "indx.store.qdrnt:QdrantStore")

    with pytest.raises(ModuleNotFoundError) as exc:
        registry._resolve("store", "qdrant")

    assert not isinstance(exc.value, MissingExtraError)
    assert exc.value.name == "indx.store.qdrnt"
