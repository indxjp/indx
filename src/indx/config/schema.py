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

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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

    def options(self, backend: str | None = None) -> dict[str, Any]:
        """Backend-specific kwargs to forward to the adapter constructor.

        Merges any top-level passthrough keys on this slot with the matching backend
        sub-table (``[<slot>.<backend>]``). Selector fields and nested sub-tables for
        *other* backends are excluded. When ``backend`` is omitted the slot's own
        selector value is used.
        """
        selector = backend if backend is not None else self._active_backend()
        out: dict[str, Any] = {}
        for key, value in self._iter_passthrough():
            if isinstance(value, dict):
                continue  # nested sub-tables are handled below, by name
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

    def _iter_passthrough(self) -> list[tuple[str, Any]]:
        data = self.model_dump()
        return [(k, v) for k, v in data.items() if k not in self._selector_fields]

    def _active_backend(self) -> str | None:
        for name in self._selector_fields:
            value = getattr(self, name, None)
            if isinstance(value, str):
                return value
        return None


class ParserConfig(_SlotConfig):
    _selector_fields = ("engine",)
    engine: str = DEFAULT_PARSER


class EnrichConfig(_SlotConfig):
    # The enrich slot carries two selectors (llm + vlm) and the metadata switch; none of
    # them are adapter passthrough kwargs.
    _selector_fields = ("llm", "vlm", "metadata")
    llm: str = DEFAULT_LLM
    vlm: str = DEFAULT_VLM
    metadata: list[str] = Field(default_factory=lambda: list(_DEFAULT_METADATA))


class EmbedConfig(_SlotConfig):
    _selector_fields = ("model",)
    model: str = DEFAULT_EMBEDDER


class StoreConfig(_SlotConfig):
    _selector_fields = ("backend",)
    backend: str = DEFAULT_STORE


class OutputConfig(_SlotConfig):
    _selector_fields = ("format",)
    format: str = DEFAULT_FORMAT


class Config(BaseModel):
    """The fully-resolved configuration the pipeline runs from."""

    model_config = ConfigDict(extra="allow")

    parser: ParserConfig = Field(default_factory=ParserConfig)
    enrich: EnrichConfig = Field(default_factory=EnrichConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

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
