"""Regression: DirectoryPipeline() with no args surfaces an offline-mode hint.

When the default (cloud-backed) stack raises MissingExtraError because the required
extra is not installed, and the caller has not explicitly set any slot, the error's
__notes__ should guide the user to the zero-dependency offline invocation (F4).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from indx.errors import MissingExtraError
from indx.pipeline import DirectoryPipeline


def _raise_missing(*args: object, **kwargs: object) -> None:
    raise MissingExtraError("parser", "docling", "docling")


def test_no_args_missing_extra_has_offline_hint() -> None:
    """MissingExtraError raised on the zero-arg path carries an offline hint note."""
    with (
        patch("indx.pipeline.pipeline.get_parser", side_effect=_raise_missing),
        pytest.raises(MissingExtraError) as exc_info,
    ):
        DirectoryPipeline()

    exc = exc_info.value
    notes = "\n".join(getattr(exc, "__notes__", []))
    assert "plaintext" in notes or "offline" in notes, (
        f"Expected offline hint in __notes__, got: {notes!r}"
    )


def test_explicit_slot_missing_extra_has_no_hint() -> None:
    """When a slot was explicitly set, no offline hint is appended (the user already chose)."""
    with (
        patch("indx.pipeline.pipeline.get_parser", side_effect=_raise_missing),
        pytest.raises(MissingExtraError) as exc_info,
    ):
        DirectoryPipeline(parser="docling")

    exc = exc_info.value
    notes = getattr(exc, "__notes__", [])
    assert not any("plaintext" in n or "offline" in n for n in notes), (
        f"Unexpected offline hint for explicit slot choice; __notes__: {notes!r}"
    )
