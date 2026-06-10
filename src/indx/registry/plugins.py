"""Discover third-party slot implementations via ``importlib.metadata`` entry points.

A package extends indx without modifying it by advertising entry points under groups like
``indx.parsers`` / ``indx.stores`` / ``indx.embedders`` (file-architecture §6). The registry
consults these only after its first-party builtins, so builtins always win name collisions;
a plugin may not silently shadow one.
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

from indx.errors import RegistryError
from indx.registry.builtins import STAGE_ENTRY_POINT_GROUP


def discover(group: str) -> dict[str, str]:
    """Return ``{name: "module:attr"}`` for every entry point advertised under ``group``."""
    found: dict[str, str] = {}
    for ep in md.entry_points(group=group):
        found.setdefault(ep.name, ep.value)
    return found


def load_plugin(group: str, name: str) -> Any | None:
    """Load the callable registered as ``name`` under ``group``, or ``None`` if absent.

    A class or a factory function are both valid: the registry calls the returned object
    with ``**kwargs``, so any callable works. A matched-but-non-callable entry point is a
    plugin-author mistake (it points at a module/constant instead of a class/factory), so it
    raises :class:`RegistryError` naming the entry point rather than returning ``None`` --
    that keeps it distinguishable from "name absent" instead of degrading into a misleading
    "unknown" error at the call site.
    """
    for ep in md.entry_points(group=group):
        if ep.name == name:
            obj = ep.load()
            if not callable(obj):
                raise RegistryError(f"entry point {group}:{name} = {ep.value!r} is not callable")
            return obj
    return None


def discover_stages() -> dict[str, Any]:
    """Return ``{name: stage-factory}`` for every third-party stage under ``indx.stages``.

    Mirrors the component entry-point discovery (file-architecture §6) for the pipeline
    ``stages`` slot documented in registry-and-defaults. Each value is the loaded entry-point
    object: a ``Stage`` class, or any zero-arg callable returning a ``Stage`` instance. The
    PipelineAPI layer instantiates/inserts these; discovery here stays construction-free and
    lazy so importing the registry never pays for a plugin's heavy deps. First duplicate name
    wins, matching :func:`discover`.
    """
    found: dict[str, Any] = {}
    for ep in md.entry_points(group=STAGE_ENTRY_POINT_GROUP):
        if ep.name not in found:
            found[ep.name] = ep.load()
    return found
