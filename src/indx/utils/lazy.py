"""Friendly missing-dependency errors for optional extras."""

from __future__ import annotations

import importlib.util

from indx.errors import MissingExtraError


def require_extra(slot: str, name: str, extra: str, *module_names: str) -> None:
    """Raise :class:`MissingExtraError` if any required module is not importable.

    Called at the top of an extra-gated implementation's ``__init__``/import so selecting a
    slot whose extra is absent fails with a single actionable message instead of a raw
    ``ImportError`` deep in a third-party package (file-architecture §5).
    """
    for module in module_names:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            # A dotted name (e.g. "azure.search.documents") whose *parent* package is
            # absent makes find_spec raise rather than return None; treat that as missing
            # too so namespace-package backends gate cleanly on a fresh install.
            raise MissingExtraError(slot=slot, name=name, extra=extra) from None
        if spec is None:
            raise MissingExtraError(slot=slot, name=name, extra=extra)
