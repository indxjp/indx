"""Canonical default slot selections — the single source of truth.

Both the pydantic field defaults in :mod:`indx.config.schema` and the
:class:`~indx.pipeline.pipeline.DirectoryPipeline` constructor fall back to these
names when a slot is left unset, so the two surfaces can never drift. These are the
documented zero-config cloud stack (technical-spec §8.2 / sdk.md "Constructor").
"""

from __future__ import annotations

from typing import Final

DEFAULT_PARSER: Final = "docling"
DEFAULT_LLM: Final = "openai:gpt-5-mini"
DEFAULT_VLM: Final = "none"
DEFAULT_EMBEDDER: Final = "openai:text-embedding-3-small"
DEFAULT_STORE: Final = "qdrant"
DEFAULT_FORMAT: Final = ".indx"

__all__ = [
    "DEFAULT_PARSER",
    "DEFAULT_LLM",
    "DEFAULT_VLM",
    "DEFAULT_EMBEDDER",
    "DEFAULT_STORE",
    "DEFAULT_FORMAT",
]
