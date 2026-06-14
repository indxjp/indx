"""Resolve a slot name to an instantiated implementation.

Resolution order: first-party builtins, then third-party entry-point plugins. Built-ins
win collisions. A missing extra surfaces as :class:`MissingExtraError` only when that slot
is actually selected (file-architecture §5). Names may carry a ``:model`` suffix (e.g.
``openai:gpt-5-mini``), which is split off and passed to the implementation as ``model=``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, cast

from indx.errors import MissingExtraError, RegistryError
from indx.registry.builtins import BUILTINS, ENTRY_POINT_GROUPS, EXTRAS
from indx.registry.plugins import discover, load_plugin


def _load(target: str) -> type:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return cast(type, getattr(module, attr))


def _resolve(slot: str, name: str) -> type:
    builtin = BUILTINS.get(slot, {}).get(name)
    if builtin is not None:
        adapter_module = builtin.partition(":")[0]
        try:
            return _load(builtin)
        except MissingExtraError:
            raise
        except ModuleNotFoundError as exc:
            # Only a genuinely-absent optional vendor dep should surface as a missing-extra
            # hint. A broken BUILTINS path (typo/post-refactor rename) makes the adapter
            # module itself unimportable; that is a code/packaging bug no extra can fix, so
            # let it propagate raw instead of conflating it with an absent extra.
            if exc.name == adapter_module or adapter_module.startswith(f"{exc.name}."):
                raise
            extra = EXTRAS.get((slot, name), name)
            raise MissingExtraError(slot=slot, name=name, extra=extra) from exc

    for group in ENTRY_POINT_GROUPS.get(slot, ()):
        plugin = load_plugin(group, name)
        if plugin is not None:
            # load_plugin is typed Any | None (a class or factory callable); the registry
            # contract is a type/constructor, so pin it at the edge.
            return cast(type, plugin)

    # Fold installed third-party plugin names into the diagnostic so a user who typo'd the
    # name of a plugin they actually installed sees it listed alongside the builtins.
    known_names = set(BUILTINS.get(slot, {}))
    for group in ENTRY_POINT_GROUPS.get(slot, ()):
        known_names.update(discover(group))
    known = sorted(known_names)
    raise RegistryError(f"unknown {slot} '{name}'. Known {slot}s: {known}")


def _accepts_model(cls: type) -> bool:
    """Whether ``cls.__init__`` can take a ``model=`` kwarg.

    True if the constructor declares an explicit ``model`` parameter or a ``**kwargs``
    catch-all. Builtins (e.g. ``object.__init__``) have no introspectable signature; treat
    those as accepting it so the call proceeds and any genuine mismatch still surfaces.
    """
    try:
        params = inspect.signature(cls.__init__).parameters  # type: ignore[misc]
    except (ValueError, TypeError):
        return True
    return any(
        p.name == "model" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _make(slot: str, spec: str, **kwargs: Any) -> Any:
    """Instantiate ``slot`` from a name that may carry a ``:model`` suffix."""
    base, sep, model = spec.partition(":")
    cls = _resolve(slot, base)
    if sep and model and "model" not in kwargs:
        # Only model-bearing slots (llm/embedder/vlm) accept the suffix; stores/writers/
        # parsers take no ``model`` kwarg, so a suffix on those is a user error. Raise a
        # typed RegistryError instead of letting a raw TypeError escape the constructor.
        if not _accepts_model(cls):
            raise RegistryError(
                f"{slot} '{base}' does not accept a ':model' suffix (got '{spec}'); "
                f"the '{slot}' slot takes no model argument"
            )
        kwargs["model"] = model
    return cls(**kwargs)


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
