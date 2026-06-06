"""T1 — registry resolution and friendly missing-extra / unknown-name errors."""

from __future__ import annotations

import pytest

from indx.errors import MissingExtraError, RegistryError
from indx.registry import get_embedder, get_parser, get_store


def test_resolves_builtins() -> None:
    assert get_parser("plaintext").name == "plaintext"
    assert get_store("jsonl").name == "jsonl"
    assert get_embedder("hash").dim == 256


def test_unknown_name_raises_registry_error() -> None:
    with pytest.raises(RegistryError, match="unknown store"):
        get_store("does-not-exist")


def test_missing_extra_message_names_pip_target() -> None:
    # qdrant is registered but its extra is not installed in the test env.
    with pytest.raises(MissingExtraError) as exc:
        get_store("qdrant")
    assert "pip install indx[qdrant]" in str(exc.value)
