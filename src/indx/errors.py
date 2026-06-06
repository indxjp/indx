"""Shared exception hierarchy for indx.

All indx-raised errors subclass :class:`IndxError` so callers (and the CLI) can catch the
whole family in one place. Messages are written to be *actionable*: they name the failing
slot, stage, or file (NFR-OBS-1).
"""

from __future__ import annotations


class IndxError(Exception):
    """Base class for every error indx raises deliberately."""


class MissingExtraError(IndxError):
    """A selected slot implementation needs an optional extra that isn't installed.

    The message always names the exact ``pip install indx[...]`` to run (file-architecture §5).
    """

    def __init__(self, slot: str, name: str, extra: str) -> None:
        self.slot = slot
        self.name = name
        self.extra = extra
        super().__init__(f"{slot} '{name}' requires the '{extra}' extra: pip install indx[{extra}]")


class MissingDependencyError(MissingExtraError):
    """Alias of :class:`MissingExtraError` (errors-and-exit-codes.md).

    Kept as a subclass so both names resolve to the same family and the existing
    ``MissingExtraError`` exit-code mapping continues to catch it.
    """


class RegistryError(IndxError):
    """A slot name could not be resolved to an implementation."""


class ConfigError(IndxError):
    """An ``indx.toml`` file (or an inline override) was missing, malformed, or invalid."""


class CredentialsError(ConfigError):
    """A vendor adapter could not authenticate or connect (missing API key, refused host).

    Subclasses :class:`ConfigError` so it maps to the configuration exit code (3) and is
    caught by the existing ``ConfigError`` handler: a missing credential is a configuration
    problem, not a runtime crash. The message names the slot and the env var to set so the
    user gets an actionable fix instead of a raw vendor traceback (NFR-OBS-1).
    """


class StageError(IndxError):
    """A pipeline stage failed. Names the stage and (when known) the offending file."""

    def __init__(self, stage: str, message: str, *, path: str | None = None) -> None:
        self.stage = stage
        self.path = path
        where = f" [{path}]" if path else ""
        super().__init__(f"stage '{stage}' failed{where}: {message}")


class ArchiveError(IndxError):
    """A .indx archive could not be read/written or failed a version/integrity check."""


class ParseError(IndxError):
    """A Parser slot failed to parse a source file (errors-and-exit-codes.md)."""


class EmbedError(IndxError):
    """An Embedder slot failed to produce vectors (errors-and-exit-codes.md)."""


class StoreError(IndxError):
    """A vector Store backend failed an upsert/search/delete (errors-and-exit-codes.md)."""


class PipelineError(IndxError):
    """A pipeline run failed outside of a single stage (errors-and-exit-codes.md)."""
