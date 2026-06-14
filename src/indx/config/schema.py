"""Pydantic v2 models for the full ``indx.toml`` schema + defaults (technical-spec §9).

Defaults are the documented zero-config stack. Adapter sub-tables (e.g. ``[store.qdrant]``)
are accepted verbatim via ``extra="allow"`` and passed opaquely to the adapter constructor.

Each slot model exposes an ``options`` mapping that surfaces only its backend-specific
sub-table (e.g. ``[store.qdrant]``) plus any other passthrough keys, so the registry /
pipeline can forward them as kwargs to the adapter constructor (configuration reference
"Backend-specific sub-tables"). The selector field itself (``engine`` / ``backend`` /
``model`` / ``format``) is never included in ``options``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from indx.pipeline.filters import WalkFilter

from indx.config.defaults import (
    DEFAULT_EMBEDDER,
    DEFAULT_FORMAT,
    DEFAULT_LLM,
    DEFAULT_PARSER,
    DEFAULT_STORE,
    DEFAULT_VLM,
)

_DEFAULT_METADATA = ["type", "topics", "tags", "summary"]


class _SlotConfig(BaseModel):
    """Base for a single pipeline-slot config table.

    ``extra="allow"`` keeps backend sub-tables (``[store.qdrant]``) and any other
    passthrough keys on the model; :meth:`options` exposes them as adapter kwargs.
    """

    model_config = ConfigDict(extra="allow")

    # Field names that are slot *selectors*, not adapter passthrough options. Subclasses
    # override this; everything else on the model becomes an adapter constructor kwarg.
    _selector_fields: tuple[str, ...] = ()

    # Registry slot name(s) whose registered backend names label this slot's sub-tables
    # (``[<slot>.<backend>]``). Used to tell a backend sub-table apart from a dict-valued
    # passthrough option. Subclasses override; ``enrich`` spans both ``llm`` and ``vlm``.
    _registry_slots: tuple[str, ...] = ()

    def options(self, backend: str | None = None) -> dict[str, Any]:
        """Backend-specific kwargs to forward to the adapter constructor.

        Merges any top-level passthrough keys on this slot with the matching backend
        sub-table (``[<slot>.<backend>]``). Selector fields and nested sub-tables for
        *other* backends are excluded. When ``backend`` is omitted the slot's own
        selector value is used.
        """
        selector = backend if backend is not None else self._active_backend()
        backend_names = self._backend_names()
        out: dict[str, Any] = {}
        for key, value in self._iter_passthrough():
            # A top-level dict is a backend sub-table (``[<slot>.<backend>]``) only when its
            # key names a backend this slot can select. The active backend's sub-table is
            # merged below; sibling backends' sub-tables are dropped. Every other dict is a
            # legitimate dict-valued passthrough option (e.g. ``headers``) and is forwarded.
            if isinstance(value, dict) and key in backend_names:
                continue
            out[key] = value
        if selector is not None:
            sub = self._sub_tables().get(selector)
            if isinstance(sub, dict):
                out.update(sub)
        return out

    def sub_tables(self) -> dict[str, dict[str, Any]]:
        """All captured backend sub-tables on this slot, keyed by backend name."""
        return dict(self._sub_tables())

    def _sub_tables(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self._iter_passthrough() if isinstance(v, dict)}

    def _backend_names(self) -> set[str]:
        """Backend names this slot can select, used to tell a backend sub-table
        (``[<slot>.<backend>]``) apart from a dict-valued passthrough option.

        Sourced from the registry's static built-in table (no heavy imports) so *every*
        registered backend for this slot — not just the active one — is recognised as a
        sub-table key and never leaks into the adapter kwargs. The active selector's own
        backend is always included so an explicitly-passed ``backend`` still resolves even
        if it is not a built-in. ``enrich`` spans both its ``llm`` and ``vlm`` slots.
        """
        from indx.registry.builtins import BUILTINS

        names: set[str] = set()
        for slot in self._registry_slots:
            names.update(BUILTINS.get(slot, {}))
        for field in self._selector_fields:
            value = getattr(self, field, None)
            if isinstance(value, str) and value:
                names.add(value.split(":", 1)[0])
        return names

    def _iter_passthrough(self) -> list[tuple[str, Any]]:
        data = self.model_dump()
        return [(k, v) for k, v in data.items() if k not in self._selector_fields]

    def _active_backend(self) -> str | None:
        for name in self._selector_fields:
            value = getattr(self, name, None)
            if isinstance(value, str) and value:
                return value.split(":", 1)[0]
        return None


class ParserConfig(_SlotConfig):
    _selector_fields = ("engine",)
    _registry_slots = ("parser",)
    engine: str = DEFAULT_PARSER


class EnrichConfig(_SlotConfig):
    # The enrich slot carries two selectors (llm + vlm) and the metadata switch; none of
    # them are adapter passthrough kwargs.
    _selector_fields = ("llm", "vlm", "metadata")
    _registry_slots = ("llm", "vlm")
    llm: str = DEFAULT_LLM
    vlm: str = DEFAULT_VLM
    metadata: list[str] = Field(default_factory=lambda: list(_DEFAULT_METADATA))


class EmbedConfig(_SlotConfig):
    _selector_fields = ("model",)
    _registry_slots = ("embedder",)
    model: str = DEFAULT_EMBEDDER


class StoreConfig(_SlotConfig):
    _selector_fields = ("backend",)
    _registry_slots = ("store",)
    backend: str = DEFAULT_STORE


class OutputConfig(_SlotConfig):
    _selector_fields = ("format",)
    _registry_slots = ("writer",)
    format: str = DEFAULT_FORMAT


class WalkConfig(BaseModel):
    """The ``[walk]`` section of indx.toml — the build-time file filter (Feature 5).

    Mirrors WalkFilter's fields; resolved into a WalkFilter by the build command. ``extra``
    is forbidden so a typo (``[walk] exclud = ...``) is a ConfigError, not a silent no-op.
    ``min_size``/``max_size`` are typed ``int | str | None`` so a TOML ``max_size = "2mb"``
    string survives validation here and the suffix is parsed once, centrally, by WalkFilter.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    ext: list[str] = Field(default_factory=list)
    name: list[str] = Field(default_factory=list)
    min_size: int | str | None = None
    max_size: int | str | None = None
    max_files: int | None = None
    max_depth: int | None = None

    def to_filter(self) -> WalkFilter:
        """Build a WalkFilter from this section (size suffix parsing happens in WalkFilter)."""
        # Local import avoids a schema→pipeline import cycle (the safe rule this repo follows
        # for cross-layer references); WalkFilter itself imports only indx.errors.
        from indx.pipeline.filters import WalkFilter

        return WalkFilter(**self.model_dump())


class Config(BaseModel):
    """The fully-resolved configuration the pipeline runs from."""

    model_config = ConfigDict(extra="allow")

    parser: ParserConfig = Field(default_factory=ParserConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    walk: WalkConfig = Field(default_factory=WalkConfig)

    def slot_options(self) -> dict[str, dict[str, Any]]:
        """Per-slot adapter kwargs, keyed by slot name (``parser``/``store``/...).

        The pipeline/registry can forward each entry as ``**kwargs`` to the matching
        adapter constructor. Backend sub-tables (e.g. ``[store.qdrant]``) are resolved
        against each slot's active selector. The ``enrich`` slot is split into the two
        adapter slots it actually drives — ``llm`` and ``vlm`` — each scoped to its own
        selector so ``[enrich.openai]`` / ``[enrich.<vlm>]`` reach the right adapter.
        """
        return {
            "parser": self.parser.options(),
            "llm": self.enrich.options(self.enrich.llm.split(":", 1)[0]),
            "vlm": self.enrich.options(self.enrich.vlm.split(":", 1)[0]),
            "embedder": self.embed.options(),
            "store": self.store.options(),
            "output": self.output.options(),
        }
