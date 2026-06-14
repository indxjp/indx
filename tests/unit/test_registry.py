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


def test_model_suffix_on_non_model_slot_raises_registry_error() -> None:
    # Stores take no ``model`` kwarg, so a ``:model`` suffix must surface as a typed
    # RegistryError (an IndxError the CLI can map to an exit code), not a raw TypeError
    # escaping the constructor.
    with pytest.raises(RegistryError, match="does not accept a ':model' suffix") as exc:
        get_store("jsonl:foo")
    assert not isinstance(exc.value, TypeError)


def test_model_suffix_still_passed_to_model_slots() -> None:
    # Model-bearing slots (llm/vlm/embedder) must still receive the suffix as ``model=``.
    from indx.registry import get_llm, get_vlm

    assert get_llm("none:some-model").model == "some-model"
    assert get_vlm("none:vmodel").model == "vmodel"
    # And a bare name on a non-model slot remains unaffected.
    assert get_store("jsonl").name == "jsonl"
