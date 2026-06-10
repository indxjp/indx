"""06 Embed+Pack — embed chunks, upsert to the store, finalize the manifest (FR-S06)."""

from __future__ import annotations

from indx.core.context import SpaceContext
from indx.core.source import Source
from indx.embed.base import Embedder
from indx.store.base import VectorStore


def stamp_sources(ctx: SpaceContext) -> None:
    """Stamp Source(path, folder, type) onto each chunk by matching doc_id."""
    chunks = ctx.space.chunks
    for doc in ctx.space.documents_:
        src = Source(path=doc.path, folder=doc.folder, type=doc.doc_type)
        for chunk in chunks:
            if chunk.doc_id == doc.id:
                chunk.source = src


class PackStage:
    name = "embed-pack"

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def run(self, ctx: SpaceContext) -> SpaceContext:
        # Stamp provenance onto each chunk so ``hit.source`` (a shorthand for
        # ``hit.chunk.source``) carries path/folder/type — the documented SDK surface
        # (sdk.md §search, data-models.md §Source). Runs after enrich so ``doc_type`` is final.
        stamp_sources(ctx)
        chunks = ctx.space.chunks
        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            for chunk, vec in zip(chunks, vectors, strict=True):
                chunk.embedding = vec
            self.store.upsert(chunks)
        ctx.space.manifest.embedding_model = self.embedder.name
        ctx.space.manifest.embedding_dim = self.embedder.dim
        return ctx
