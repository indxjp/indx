"""KnowledgeSpace — the root aggregate produced by processing a directory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from indx._version import __version__
from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.relation import Relation
from indx.core.stats import SpaceStats

if TYPE_CHECKING:
    from indx.store.base import SearchHit


class Manifest(BaseModel):
    """Reproducibility record embedded in every space / .indx archive (FR-CFG-3, NFR-DET-1)."""

    schema_version: str = "1"
    indx_version: str = __version__
    source_root: str = ""
    components: dict[str, str] = Field(default_factory=dict)  # slot -> name (parser, embedder, ...)
    embedding_model: str | None = None
    embedding_dim: int | None = None


class KnowledgeSpace(BaseModel):
    """Structure + relationships + semantic metadata for a directory (FR-DM-1).

    Serializes to / from a portable ``.indx`` archive via :mod:`indx.archive`.

    ``documents`` is both serialized data and a filtering accessor (data-models.md §
    KnowledgeSpace, sdk.md): the document list is stored as the field ``documents_`` (with
    the serialization alias ``documents`` so existing ``KnowledgeSpace(documents=[...])``
    construction and on-disk shape keep working), and the public name ``documents`` resolves
    to the callable :meth:`documents` method below.
    """

    model_config = ConfigDict(populate_by_name=True)

    manifest: Manifest = Field(default_factory=Manifest)
    documents_: list[Document] = Field(default_factory=list, alias="documents")
    chunks: list[Chunk] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def document(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents_ if d.id == doc_id), None)

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    # ------------------------------------------------------------------ accessors

    def documents(self, type: str | None = None) -> list[Document]:  # noqa: A002
        """Return the space's documents, optionally filtered to a detected ``type``.

        ``space.documents()`` returns every document; ``space.documents(type="policy")``
        narrows to one detected type (sdk.md §documents, data-models.md §KnowledgeSpace).
        """
        if type is None:
            return list(self.documents_)
        return [d for d in self.documents_ if d.doc_type == type]

    @property
    def stats(self) -> SpaceStats:
        """Aggregate counts for the space (sdk.md §stats, data-models.md §SpaceStats)."""
        from collections import Counter

        types: Counter[str] = Counter(d.doc_type or "unknown" for d in self.documents_)
        embeddings = sum(1 for c in self.chunks if c.embedding is not None)
        return SpaceStats(
            documents=len(self.documents_),
            chunks=len(self.chunks),
            relations=len(self.relations),
            embeddings=embeddings,
            embed_dim=self.manifest.embedding_dim,
            types=dict(types),
            bytes_source=sum(d.size_bytes for d in self.documents_),
        )

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Embed ``query`` and return the top-``k`` chunks as :class:`SearchHit`s.

        Mirrors ``indx query``: the embedder is resolved from the manifest (falling back to
        ``hash`` for the offline core), a self-contained ``jsonl`` store is rebuilt from the
        space's chunks, and each hit's neighbors are resolved into full
        :class:`~indx.core.chunk.Chunk` objects (sdk.md §search).
        """
        from indx.registry import get_embedder, get_store

        m = self.manifest
        embedder_name = m.components.get("embedder") or m.embedding_model or "hash"
        embedder = get_embedder(embedder_name)

        store = get_store("jsonl")
        store.upsert(self.chunks)
        vector = embedder.embed([query])[0]
        hits = store.search(vector, k=k)

        by_id = {c.id: c for c in self.chunks}
        for hit in hits:
            hit.neighbors = [by_id[n] for n in hit.chunk.neighbors if n in by_id]
        return list(hits)

    # ------------------------------------------------------------------ persistence

    @classmethod
    def load(cls, archive: str) -> KnowledgeSpace:
        """Open a sealed ``.indx`` archive into a :class:`KnowledgeSpace` (sdk.md §load)."""
        from pathlib import Path

        from indx.archive import read_archive

        return read_archive(Path(archive))

    def save(self, archive: str) -> None:
        """Seal this space into a portable ``.indx`` archive (sdk.md §save)."""
        from pathlib import Path

        from indx.archive import write_archive

        write_archive(self, Path(archive))
