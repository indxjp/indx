"""Embedder slot. Default = bge-m3 (extra); a deterministic hash embedder ships in core.

Re-exports the ``Embedder`` protocol so ``from indx.embed import Embedder`` works as the
SDK surface documents (technical-spec §8.1).
"""

from indx.embed.base import Embedder

__all__ = ["Embedder"]
