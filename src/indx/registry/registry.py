"""Resolve a slot name to an instantiated implementation.

Resolution order: first-party builtins, then third-party entry-point plugins. Built-ins
win collisions. A missing extra surfaces as :class:`MissingExtraError` only when that slot
is actually selected (file-architecture §5). Names may carry a ``:model`` suffix (e.g.
``openai:gpt-5-mini``), which is split off and passed to the implementation as ``model=``.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from indx.errors import MissingExtraError, RegistryError
from indx.registry.builtins import BUILTINS, ENTRY_POINT_GROUPS, EXTRAS
from indx.registry.plugins import load_plugin


def _load(target: str) -> type:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return cast(type, getattr(module, attr))


def _resolve(slot: str, name: str) -> type:
    builtin = BUILTINS.get(slot, {}).get(name)
    if builtin is not None:
        try:
            return _load(builtin)
        except MissingExtraError:
            raise
        except ModuleNotFoundError as exc:  # the extra's transitive dep is absent
            extra = EXTRAS.get((slot, name), name)
            raise MissingExtraError(slot=slot, name=name, extra=extra) from exc

    for group in ENTRY_POINT_GROUPS.get(slot, ()):
        plugin = load_plugin(group, name)
        if plugin is not None:
            return plugin

    known = sorted(BUILTINS.get(slot, {}))
    raise RegistryError(f"unknown {slot} '{name}'. Known {slot}s: {known}")


def _make(slot: str, spec: str, **kwargs: Any) -> Any:
    """Instantiate ``slot`` from a name that may carry a ``:model`` suffix."""
    base, sep, model = spec.partition(":")
    if sep and model and "model" not in kwargs:
        kwargs["model"] = model
    return _resolve(slot, base)(**kwargs)


def get_parser(name: str = "docling", **kwargs: Any) -> Any:
    return _make("parser", name, **kwargs)


def get_llm(name: str = "openai:gpt-5-mini", **kwargs: Any) -> Any:
    return _make("llm", name, **kwargs)


def get_vlm(name: str = "none", **kwargs: Any) -> Any:
    return _make("vlm", name, **kwargs)


def get_embedder(name: str = "openai:text-embedding-3-small", **kwargs: Any) -> Any:
    return _make("embedder", name, **kwargs)


def get_store(name: str = "qdrant", **kwargs: Any) -> Any:
    return _make("store", name, **kwargs)


def get_writer(name: str = "indx", **kwargs: Any) -> Any:
    return _make("writer", name, **kwargs)
