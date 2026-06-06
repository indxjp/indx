"""Find, parse, validate, and merge ``indx.toml`` into a :class:`Config`.

Precedence (configuration reference "Precedence"; technical-spec §9.2), lowest first::

    documented default  <  indx.toml  <  INDX_* env vars  <  CLI override

(Explicit code arguments / ``use()`` sit above the CLI but are applied by the SDK, not
here.) Env layering uses ``pydantic-settings`` with the documented nested convention —
``INDX_<SLOT>__<KEY>`` where ``__`` separates the *slot* prefix from the key
(``INDX_STORE__BACKEND``, ``INDX_LLM__API_KEY``). The env prefix is keyed on the **slot
name**, not the ``indx.toml`` section heading, so the LLM slot (configured under
``[enrich]``) takes its env from ``INDX_LLM__*``.

A missing explicit config path or malformed/invalid TOML raises
:class:`~indx.errors.ConfigError` (CLI exit code 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python 3.11+ ships tomllib; tomli is the <3.11 fallback (pyproject extra).
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - only on <3.11
    import tomli as _toml  # type: ignore[import-not-found, no-redef]

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from indx.config.schema import (
    Config,
    EmbedConfig,
    EnrichConfig,
    OutputConfig,
    ParserConfig,
    StoreConfig,
)
from indx.errors import ConfigError

_CONFIG_NAME = "indx.toml"

# CLI override key -> dotted path into the Config tree.
_OVERRIDE_MAP = {
    "parser": ("parser", "engine"),
    "llm": ("enrich", "llm"),
    "vlm": ("enrich", "vlm"),
    "embedder": ("embed", "model"),
    "store": ("store", "backend"),
    "format": ("output", "format"),
}

# Env slot prefix -> (Config section, selector key for that slot's own value). The env
# prefix is keyed on the slot name (configuration reference "Secrets via environment
# variables"), which is why llm/vlm both live under the [enrich] section.
_SLOT_TO_SECTION = {
    "parser": ("parser", "engine"),
    "llm": ("enrich", "llm"),
    "vlm": ("enrich", "vlm"),
    "embedder": ("embed", "model"),
    "store": ("store", "backend"),
    "output": ("output", "format"),
}

# Schema defaults for each slot's selector, used to resolve the backend a slot's env
# passthrough belongs to when neither env nor TOML pins the selector. Mirrors the field
# defaults on the matching schema model so env namespacing matches the TOML shape.
_SELECTOR_DEFAULTS = {
    "parser": ParserConfig.model_fields["engine"].default,
    "llm": EnrichConfig.model_fields["llm"].default,
    "vlm": EnrichConfig.model_fields["vlm"].default,
    "embedder": EmbedConfig.model_fields["model"].default,
    "store": StoreConfig.model_fields["backend"].default,
    "output": OutputConfig.model_fields["format"].default,
}

# Sections whose env passthrough must be namespaced under the slot's backend sub-table
# (mirroring TOML ``[enrich.<backend>]``) because the section hosts more than one adapter
# slot. Without this, INDX_LLM__* and INDX_VLM__* collide on a single shared dict.
_NAMESPACED_SLOTS = {"llm", "vlm"}


class _SlotEnv(BaseModel):
    """A single slot's env-sourced values: the selector plus arbitrary passthrough keys."""

    model_config = ConfigDict(extra="allow")


class _EnvSettings(BaseSettings):
    """Reads ``INDX_<SLOT>__<KEY>`` env vars into per-slot dicts.

    Pure env source — the TOML file is layered in :func:`load_config`, not here, so the
    documented precedence (defaults < toml < env < CLI) stays explicit and auditable.
    """

    model_config = SettingsConfigDict(
        env_prefix="INDX_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    parser: _SlotEnv = Field(default_factory=_SlotEnv)
    llm: _SlotEnv = Field(default_factory=_SlotEnv)
    vlm: _SlotEnv = Field(default_factory=_SlotEnv)
    embedder: _SlotEnv = Field(default_factory=_SlotEnv)
    store: _SlotEnv = Field(default_factory=_SlotEnv)
    output: _SlotEnv = Field(default_factory=_SlotEnv)


def find_config(path: str | Path | None = None, *, cwd: str | Path | None = None) -> Path | None:
    """Resolve the config file to use, or ``None`` if no config applies.

    An explicit ``path`` that does not exist is a :class:`ConfigError`. With no explicit
    path, ``./indx.toml`` is used when present, otherwise ``None`` (pure defaults).
    """
    if path is not None:
        resolved = Path(path)
        if not resolved.is_file():
            raise ConfigError(f"config file not found: {resolved}")
        return resolved
    candidate = Path(cwd or ".") / _CONFIG_NAME
    return candidate if candidate.is_file() else None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins), returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _slot_backend(
    slot: str, selector: str, env_section: dict[str, Any], raw: dict[str, Any]
) -> str:
    """Resolve the backend a slot's env passthrough belongs to (``env > toml > default``).

    The selector value is ``backend[:model]`` (e.g. ``openai:gpt-5-mini``); only the
    backend prefix names the sub-table. Precedence mirrors the overall env-over-file merge
    so passthrough lands under the same backend the resolved selector will pick.
    """
    section, _ = _SLOT_TO_SECTION[slot]
    value = env_section.get(selector)
    if not isinstance(value, str):
        raw_section = raw.get(section)
        toml_value = raw_section.get(selector) if isinstance(raw_section, dict) else None
        value = toml_value if isinstance(toml_value, str) else _SELECTOR_DEFAULTS[slot]
    return value.split(":", 1)[0]


def _env_layer(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate ``INDX_<SLOT>__*`` env vars into a partial Config-shaped dict.

    Each slot's selector env var (``INDX_PARSER__ENGINE``, ``INDX_STORE__BACKEND``, ...)
    maps onto the section's selector field; all other keys are forwarded verbatim as
    passthrough adapter options on the matching section. Sections hosting multiple adapter
    slots (``enrich`` carries both ``llm`` and ``vlm``) namespace their passthrough under a
    ``[<section>.<backend>]`` sub-table — mirroring the TOML shape — so the two slots' keys
    cannot collide and never cross-leak. ``raw`` (the parsed TOML) is consulted only to
    resolve each slot's effective backend for that namespacing.
    """
    env = _EnvSettings()
    layer: dict[str, Any] = {}
    for slot, (section, selector) in _SLOT_TO_SECTION.items():
        values = getattr(env, slot).model_dump()
        if not values:
            continue
        section_dict = layer.setdefault(section, {})
        # Passthrough (non-selector) keys for a multi-slot section go into a per-backend
        # sub-table so INDX_LLM__* and INDX_VLM__* reach the right adapter independently.
        passthrough_dict = section_dict
        if slot in _NAMESPACED_SLOTS:
            backend = _slot_backend(slot, selector, values, raw)
            passthrough_dict = section_dict.setdefault(backend, {})
        for key, value in values.items():
            if key == selector:
                # The slot's own selector env (e.g. INDX_STORE__BACKEND -> store.backend)
                # is routed to the section's selector field, never the passthrough table.
                section_dict[selector] = value
            else:
                passthrough_dict[key] = value
    return layer


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, str | None] | None = None,
    cwd: str | Path | None = None,
) -> Config:
    """Resolve the effective config: defaults < ``indx.toml`` < env < CLI overrides.

    Raises:
        ConfigError: the explicit path is missing, the TOML is malformed, or a section
            fails validation.
    """
    raw: dict[str, Any] = {}
    config_path = find_config(path, cwd=cwd)
    if config_path is not None:
        try:
            raw = _toml.loads(config_path.read_text(encoding="utf-8"))
        except (_toml.TOMLDecodeError, OSError) as exc:
            raise ConfigError(f"could not read {config_path}: {exc}") from exc

    # Layer env vars above the file (below CLI). Defaults are supplied by the model itself.
    merged = _deep_merge(raw, _env_layer(raw))

    try:
        config = Config.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid {config_path or _CONFIG_NAME}: {exc}") from exc

    # CLI overrides win over everything (env + file + defaults).
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, field = _OVERRIDE_MAP[key]
        setattr(getattr(config, section), field, value)
    return config
