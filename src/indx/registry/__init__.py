"""Name -> implementation resolution for every swappable slot.

The only place allowed to import concrete implementations, and it does so lazily so a
missing extra never breaks an unrelated code path (file-architecture §3).
"""

from indx.registry.plugins import discover, discover_stages, load_plugin
from indx.registry.registry import (
    get_embedder,
    get_llm,
    get_parser,
    get_store,
    get_vlm,
    get_writer,
)

__all__ = [
    "get_parser",
    "get_llm",
    "get_vlm",
    "get_embedder",
    "get_store",
    "get_writer",
    "discover",
    "discover_stages",
    "load_plugin",
]
