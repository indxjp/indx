"""Regression: a non-callable plugin entry point raises a named RegistryError.

Previously ``load_plugin`` returned ``None`` for a matched-but-non-callable entry point,
which was indistinguishable from "name absent" at the resolver call site and degraded into
a misleading ``unknown {slot}`` error. It must now raise a RegistryError naming the entry
point so a plugin author who mis-advertises their entry point sees the real problem.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from indx.errors import RegistryError
from indx.registry import plugins as P


def test_noncallable_entry_point_raises_named_registry_error() -> None:
    ep = MagicMock()
    ep.name = "fake"
    ep.value = "some.module:CONSTANT"
    ep.load.return_value = 42  # not callable

    with (
        patch.object(P.md, "entry_points", return_value=[ep]),
        pytest.raises(RegistryError) as exc,
    ):
        P.load_plugin("indx.stores", "fake")

    message = str(exc.value)
    assert "fake" in message
    assert "some.module:CONSTANT" in message
    assert "not callable" in message


def test_callable_entry_point_is_returned() -> None:
    def factory() -> object:
        return object()

    ep = MagicMock()
    ep.name = "fake"
    ep.value = "some.module:Factory"
    ep.load.return_value = factory

    with patch.object(P.md, "entry_points", return_value=[ep]):
        assert P.load_plugin("indx.stores", "fake") is factory


def test_absent_name_returns_none() -> None:
    ep = MagicMock()
    ep.name = "other"
    ep.value = "some.module:Other"

    with patch.object(P.md, "entry_points", return_value=[ep]):
        assert P.load_plugin("indx.stores", "fake") is None
