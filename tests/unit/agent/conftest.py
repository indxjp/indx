"""Shared fixtures for the agent-connector tests.

The space is constructed directly (no file IO): two enriched documents with a few chunks,
embeddings filled by the in-core ``hash`` embedder so ``KnowledgeConnector.search`` runs the
real offline retrieval path. Vendor SDKs (langchain / agents / pydantic_ai / claude / mcp)
are absent here, so the adapter tests inject fakes into ``sys.modules`` — the same recipe the
output-writer tests use.
"""

from __future__ import annotations

import pytest

from indx.agent import KnowledgeConnector
from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.core.source import Source
from indx.registry import get_embedder


def build_space() -> KnowledgeSpace:
    """A small, enriched, embedded space exercising the documented connector surface."""
    space = KnowledgeSpace(manifest=Manifest(components={"embedder": "hash"}))
    space.documents_.extend(
        [
            Document(
                id="d1",
                path="people/remote-work.md",
                lineage=["people"],
                doc_type="policy",
                topics=["remote", "work"],
                tags=["hr"],
                summary="How remote work works at Acme.",
            ),
            Document(
                id="d2",
                path="guide/onboarding.md",
                lineage=["guide"],
                doc_type="guide",
                topics=["onboarding"],
                tags=["intro"],
                summary="Onboarding steps for new hires.",
            ),
        ]
    )
    space.chunks.extend(
        [
            Chunk(
                id="c0",
                doc_id="d1",
                position=0,
                text="Acme is remote-first; employees may work from anywhere.",
                next_id="c1",
                source=Source(path="people/remote-work.md", folder="people", type="policy"),
            ),
            Chunk(
                id="c1",
                doc_id="d1",
                position=1,
                text="Core collaboration hours overlap from 10am to 2pm.",
                prev_id="c0",
                source=Source(path="people/remote-work.md", folder="people", type="policy"),
            ),
            Chunk(
                id="c2",
                doc_id="d2",
                position=0,
                text="Onboarding: set up your laptop, accounts, and read the handbook.",
                source=Source(path="guide/onboarding.md", folder="guide", type="guide"),
            ),
        ]
    )
    embedder = get_embedder("hash")
    vectors = embedder.embed([c.text for c in space.chunks])
    for chunk, vector in zip(space.chunks, vectors, strict=True):
        chunk.embedding = vector
    space.manifest.embedding_model = "hash"
    space.manifest.embedding_dim = len(vectors[0])
    return space


@pytest.fixture
def space() -> KnowledgeSpace:
    return build_space()


@pytest.fixture
def kb(space: KnowledgeSpace) -> KnowledgeConnector:
    return KnowledgeConnector(space, name="handbook")
