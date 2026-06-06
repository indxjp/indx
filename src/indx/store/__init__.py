"""Vector store slot. Default = qdrant (extra); jsonl (no-DB) ships in core.

Re-exports the store protocol so ``from indx.store import Store`` works as the SDK surface
documents (technical-spec §8.1). ``Store`` is an alias of the internal ``VectorStore``
protocol.
"""

from indx.store.base import VectorStore

# ``Store`` is the documented public name for the ``VectorStore`` protocol.
Store = VectorStore

__all__ = ["Store", "VectorStore"]
