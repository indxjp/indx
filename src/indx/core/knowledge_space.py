"""KnowledgeSpace — the root aggregate produced by processing a directory."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from indx._version import __version__
from indx.core.chunk import Chunk
from indx.core.document import Document
from indx.core.relation import Relation
from indx.core.stats import SpaceStats
from indx.store.base import SearchHit

if TYPE_CHECKING:
    from indx.pipeline.pipeline import DirectoryPipeline, IngestResult
    from indx.store.base import VectorStore


class ChildRef(BaseModel):
    """A reference from a parent manifest to a child ``.indx`` archive (Feature 2).

    Stored as one entry of :attr:`Manifest.children`. ``ref`` is a path to the child archive:
    a relative ``ref`` resolves against the directory of the parent archive (portable — ship the
    parent plus a ``children/`` folder together); an absolute ``ref`` is honored as-is.
    ``sha256`` is an optional integrity pin over the child archive *file's raw bytes*
    (``hashlib.sha256(child.read_bytes())`` via :func:`indx.archive.format.file_sha256`); a
    mismatch on resolve raises :class:`~indx.errors.ArchiveError`.
    """

    name: str
    ref: str
    sha256: str | None = None


class Answer(BaseModel):
    """The result of :meth:`KnowledgeSpace.ask` — a synthesized/extractive answer + its sources.

    ``answer`` is either an LLM-synthesized response (when the manifest selects a real ``llm``)
    or a deterministic extractive concatenation of the retrieved chunks with ``[n]`` citations
    (the offline default, ``llm="none"``). ``hits`` are the retrieved :class:`SearchHit`s in
    score order; :attr:`sources` exposes their de-duplicated cited source paths.
    """

    question: str
    answer: str
    hits: list[SearchHit] = Field(default_factory=list)
    llm: str  # the llm name that produced `answer` ("none" => extractive)

    @property
    def sources(self) -> list[str]:
        """De-duplicated, order-preserving cited source paths (``hit.source.path``)."""
        seen: dict[str, None] = {}
        for h in self.hits:
            p = h.source.path if h.source else None
            if p and p not in seen:
                seen[p] = None
        return list(seen)


class Manifest(BaseModel):
    """Reproducibility record embedded in every space / .indx archive (FR-CFG-3, NFR-DET-1)."""

    schema_version: str = "1"
    indx_version: str = __version__
    source_root: str = ""
    components: dict[str, str] = Field(default_factory=dict)  # slot -> name (parser, embedder, ...)
    embedding_model: str | None = None
    embedding_dim: int | None = None
    # Composite-space references (Feature 2). Additive + optional: a v1 manifest simply has [].
    children: list[ChildRef] = Field(default_factory=list)


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
    # Run telemetry only — number of files dropped by the WalkFilter this build (Feature 5).
    # Never persisted (``exclude=True`` keeps it out of model_dump and therefore out of every
    # archive member) so archive byte-determinism is preserved.
    skipped_files_: int = Field(default=0, exclude=True)

    # Non-serialized runtime state (Feature 2) — never written to the archive.
    # ``_source_path`` is the archive this space was read from (base dir for relative child
    # refs); ``_children_cache`` memoizes the lazily-resolved child spaces.
    _source_path: Path | None = PrivateAttr(default=None)
    _children_cache: list[KnowledgeSpace] | None = PrivateAttr(default=None)

    def _bind_source(self, path: Path) -> None:
        """Record the archive this space was read from (Feature 2).

        The archive's directory is the base for resolving relative :class:`ChildRef` refs.
        Called by :func:`indx.archive.read_archive`. Clears any cached child resolution.
        """
        self._source_path = Path(path)
        self._children_cache = None

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
        """Aggregate counts for the space (sdk.md §stats, data-models.md §SpaceStats).

        When the space references children (Feature 2) the counts federate over
        :meth:`flatten` (self + all resolved descendants); a child-less space is unchanged.
        """
        from collections import Counter

        base = self.flatten() if self.manifest.children else self
        types: Counter[str] = Counter(d.doc_type or "unknown" for d in base.documents_)
        embeddings = sum(1 for c in base.chunks if c.embedding is not None)
        return SpaceStats(
            documents=len(base.documents_),
            chunks=len(base.chunks),
            relations=len(base.relations),
            embeddings=embeddings,
            embed_dim=base.manifest.embedding_dim,
            types=dict(types),
            bytes_source=sum(d.size_bytes for d in base.documents_),
        )

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Embed ``query`` and return the top-``k`` chunks as :class:`SearchHit`s.

        Mirrors ``indx query``: the embedder is resolved from the manifest (falling back to
        ``hash`` for the offline core), a self-contained ``jsonl`` store is rebuilt from the
        space's chunks, and each hit's neighbors are resolved into full
        :class:`~indx.core.chunk.Chunk` objects (sdk.md §search).
        """
        from indx.registry import get_embedder, get_store

        # Federate over the merged view when children exist (Feature 2): one jsonl store is
        # rebuilt across the namespaced+deduped chunks, so the store's own descending-score
        # ordering IS the global re-rank and the top-k is composite-wide.
        target = self.flatten() if self.manifest.children else self
        m = target.manifest
        embedder_name = m.components.get("embedder") or m.embedding_model or "hash"
        embedder = get_embedder(embedder_name)

        store = get_store("jsonl")
        store.upsert(target.chunks)
        vector = embedder.embed([query])[0]
        hits = store.search(vector, k=k)

        by_id = {c.id: c for c in target.chunks}
        for hit in hits:
            hit.neighbors = [by_id[n] for n in hit.chunk.neighbors if n in by_id]
        return list(hits)

    # ------------------------------------------------------------------ ask (F4)

    def ask(self, question: str, k: int = 5) -> Answer:
        """Retrieve the top-``k`` chunks for ``question`` and answer with citations (Feature 4).

        Retrieval reuses :meth:`search`. Synthesis:

        * If the manifest selects a real llm (``components["llm"]`` not in ``{"", "none"}``),
          resolve it via :func:`indx.registry.get_llm` and ask it to answer **using only** the
          retrieved chunks, citing source paths. A credential/connection failure surfaces as the
          adapter's typed error (``CredentialsError`` -> exit 3) — never silently downgraded.
        * Otherwise (``llm=none``, the offline default) build a **deterministic extractive
          answer**: concatenate the top chunks' text in hit order, each suffixed with its 1-based
          ``[n]`` citation, trailed by a numbered ``Sources`` list of ``hit.source.path``. No
          model, no network.

        ``k`` <= 0 returns an empty-answer :class:`Answer` (no crash). An empty space / no hits
        returns ``answer = "No matching content found in the space."``, ``hits=[]``, ``llm="none"``.
        """
        llm_name = self.manifest.components.get("llm") or "none"
        if k <= 0:
            return Answer(question=question, answer="", hits=[], llm=llm_name)

        hits = self.search(question, k=k)
        if not hits:
            return Answer(
                question=question,
                answer="No matching content found in the space.",
                hits=[],
                llm="none",
            )

        if llm_name and llm_name != "none":
            answer_text = self._ask_llm(question, hits, llm_name)
            return Answer(question=question, answer=answer_text, hits=hits, llm=llm_name)

        return Answer(
            question=question,
            answer=self._extractive_answer(hits),
            hits=hits,
            llm="none",
        )

    @staticmethod
    def _chunk_excerpt(text: str, budget: int = 500) -> str:
        """Collapse whitespace and truncate a chunk's text to a fixed budget (deterministic)."""
        collapsed = " ".join(text.split())
        if len(collapsed) > budget:
            return collapsed[:budget].rstrip() + "…"
        return collapsed

    def _extractive_answer(self, hits: list[SearchHit]) -> str:
        """Deterministic offline answer: cited chunk excerpts + a numbered Sources list."""
        # Build the de-duplicated source ordering (first-seen) and a path -> [n] map.
        ordered_sources: list[str] = []
        index_of: dict[str, int] = {}
        for hit in hits:
            path = hit.source.path if hit.source else None
            if path and path not in index_of:
                index_of[path] = len(ordered_sources) + 1
                ordered_sources.append(path)

        body_parts: list[str] = []
        for hit in hits:
            path = hit.source.path if hit.source else None
            cite = f" [{index_of[path]}]" if path else ""
            body_parts.append(f"{self._chunk_excerpt(hit.chunk.text)}{cite}")
        body = "\n\n".join(body_parts)

        if not ordered_sources:
            return body
        sources_block = "\n".join(
            f"[{i}] {path}" for i, path in enumerate(ordered_sources, start=1)
        )
        return f"{body}\n\nSources:\n{sources_block}"

    def _ask_llm(self, question: str, hits: list[SearchHit], llm_name: str) -> str:
        """Synthesize an answer with the configured llm, citing the numbered context."""
        from indx.registry import get_llm

        ordered_sources: list[str] = []
        index_of: dict[str, int] = {}
        for hit in hits:
            path = hit.source.path if hit.source else None
            if path and path not in index_of:
                index_of[path] = len(ordered_sources) + 1
                ordered_sources.append(path)

        context_parts: list[str] = []
        for hit in hits:
            path = hit.source.path if hit.source else None
            cite = index_of[path] if path else 0
            context_parts.append(f"[{cite}] {self._chunk_excerpt(hit.chunk.text)}")
        context = "\n\n".join(context_parts)

        system = (
            "You answer questions using ONLY the numbered context provided. "
            "Do not use outside knowledge. Cite supporting context with [n] markers that match "
            "the numbered sources. If the context does not contain the answer, say so."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        llm = get_llm(llm_name)
        return str(llm.complete(prompt, system=system, max_tokens=512, temperature=0.0))

    # ------------------------------------------------------------------ composite (F2)

    def children(self) -> list[KnowledgeSpace]:
        """Resolve + cache each :class:`ChildRef` to a loaded child space (Feature 2).

        Resolution per ``ChildRef`` (in manifest order, which is kept sorted by name):

        1. Resolve ``ref``: absolute → as-is; relative → ``self._source_path.parent / ref``.
           A relative ref on an in-memory space (never read from disk, so ``_source_path`` is
           ``None``) raises :class:`~indx.errors.ArchiveError`.
        2. A missing file → ``ArchiveError`` ("child archive not found").
        3. If ``ChildRef.sha256`` is set and the file's bytes do not match → ``ArchiveError``.
        4. :func:`indx.archive.read_archive` (verifies the child's own internal checksums +
           zip-bomb caps; a corrupt child surfaces its own ``ArchiveError``).

        Lazy + memoized: repeated calls return the same loaded child instances.
        """
        if self._children_cache is not None:
            return self._children_cache
        from indx.archive import format as fmt
        from indx.archive import read_archive
        from indx.errors import ArchiveError

        resolved: list[KnowledgeSpace] = []
        for child in self.manifest.children:
            ref_path = Path(child.ref)
            if ref_path.is_absolute():
                target = ref_path
            elif self._source_path is None:
                raise ArchiveError(
                    f"cannot resolve relative child {child.ref!r} for '{child.name}': "
                    "parent space has no archive path (it was not read from disk)"
                )
            else:
                target = self._source_path.parent / ref_path
            if not target.is_file():
                raise ArchiveError(f"child archive not found for '{child.name}': {target}")
            if child.sha256 is not None:
                got = fmt.file_sha256(target)
                if got != child.sha256:
                    raise ArchiveError(
                        f"child checksum mismatch for '{child.name}': "
                        f"expected {child.sha256}, got {got}"
                    )
            resolved.append(read_archive(target))
        self._children_cache = resolved
        return resolved

    def flatten(self) -> KnowledgeSpace:
        """Merge self + all descendants into one read-only space (Feature 2).

        Depth-first over the (name-sorted) child tree. Cycle-guarded by a visited-set of resolved
        real paths (``Path.resolve()``), mirroring :func:`indx.utils.io._iter_files`: a space whose
        real path is already on the stack is skipped, so ``A→B→A`` terminates. Descendant ids are
        namespaced by the ``ChildRef.name`` path from the root, joined by ``/`` (e.g. ``eng/d1``,
        ``eng/backend/d1``); the parent's own ids keep their bare form. Namespacing rewrites
        ``Document.id``, ``Chunk`` ids/neighbor links and ``Relation.src``/``dst``
        consistently so neighbor links and edges stay valid. Dedup is by final namespaced id,
        depth-first first-wins. The merged manifest carries ``children=[]`` so federated
        ``stats``/``search`` over it do not recurse again. Mutates nothing on disk.
        """
        docs: list[Document] = []
        chunks: list[Chunk] = []
        relations: list[Relation] = []
        seen_doc: set[str] = set()
        seen_chunk: set[str] = set()
        self._merge_into(docs, chunks, relations, seen_doc, seen_chunk, prefix="", visited=set())
        merged = KnowledgeSpace(
            manifest=self.manifest.model_copy(update={"children": []}),
            documents=docs,
            chunks=chunks,
            relations=relations,
        )
        return merged

    def _merge_into(
        self,
        docs: list[Document],
        chunks: list[Chunk],
        relations: list[Relation],
        seen_doc: set[str],
        seen_chunk: set[str],
        *,
        prefix: str,
        visited: set[Path],
    ) -> None:
        """Depth-first accumulate this space's (namespaced) rows, then recurse into children."""
        if self._source_path is not None:
            real = self._source_path.resolve()
            if real in visited:
                return
            visited.add(real)

        def ns(local_id: str) -> str:
            return f"{prefix}{local_id}" if prefix else local_id

        for d in self.documents_:
            new_id = ns(d.id)
            if new_id in seen_doc:
                continue
            seen_doc.add(new_id)
            docs.append(
                d.model_copy(
                    update={
                        "id": new_id,
                        "chunk_ids": [ns(c) for c in d.chunk_ids],
                        "references": [
                            r.model_copy(update={"src": ns(r.src), "dst": ns(r.dst)})
                            for r in d.references
                        ],
                        "referenced_by": [
                            r.model_copy(update={"src": ns(r.src), "dst": ns(r.dst)})
                            for r in d.referenced_by
                        ],
                    }
                )
            )
        for c in self.chunks:
            new_id = ns(c.id)
            if new_id in seen_chunk:
                continue
            seen_chunk.add(new_id)
            chunks.append(
                c.model_copy(
                    update={
                        "id": new_id,
                        "doc_id": ns(c.doc_id),
                        "prev_id": ns(c.prev_id) if c.prev_id is not None else None,
                        "next_id": ns(c.next_id) if c.next_id is not None else None,
                        "relations": [
                            r.model_copy(update={"src": ns(r.src), "dst": ns(r.dst)})
                            for r in c.relations
                        ],
                    }
                )
            )
        for r in self.relations:
            relations.append(r.model_copy(update={"src": ns(r.src), "dst": ns(r.dst)}))

        for child_ref, child_space in zip(self.manifest.children, self.children(), strict=True):
            child_prefix = f"{prefix}{child_ref.name}/"
            child_space._merge_into(
                docs,
                chunks,
                relations,
                seen_doc,
                seen_chunk,
                prefix=child_prefix,
                visited=visited,
            )

    def add_child(
        self, name: str, ref: str, *, sha256: str | None = None, auto_pin: bool = True
    ) -> ChildRef:
        """Add (or replace by name) a :class:`ChildRef` on the manifest (Feature 2).

        Does NOT read or mutate the child archive; only edits ``self.manifest.children``.
        Re-adding an existing name replaces it (idempotent compose). Keeps ``children`` sorted by
        name (determinism) and clears the resolve cache. When ``sha256`` is ``None`` and
        ``auto_pin`` is true and ``ref`` resolves to an existing file at call time, its digest is
        computed and pinned; ``auto_pin=False`` stores the ref unpinned (the compose ``--no-pin``
        path — see :func:`indx.cli.compose.compose_command`).
        """
        from indx.archive import format as fmt

        pin = sha256
        if pin is None and auto_pin:
            ref_path = Path(ref)
            target = (
                ref_path
                if ref_path.is_absolute() or self._source_path is None
                else self._source_path.parent / ref_path
            )
            if target.is_file():
                pin = fmt.file_sha256(target)
        stored = ChildRef(name=name, ref=ref, sha256=pin)
        kept = [c for c in self.manifest.children if c.name != name]
        kept.append(stored)
        self.manifest.children = sorted(kept, key=lambda c: c.name)
        self._children_cache = None
        return stored

    def remove_child(self, name: str) -> ChildRef | None:
        """Remove the :class:`ChildRef` with this name (Feature 2).

        Returns the removed ref, or ``None`` if absent. Clears the resolve cache.
        """
        found = next((c for c in self.manifest.children if c.name == name), None)
        if found is not None:
            self.manifest.children = [c for c in self.manifest.children if c.name != name]
            self._children_cache = None
        return found

    # ------------------------------------------------------------------ CRUD (Feature 3)

    def add(self, path: str | Path, *, pipeline: DirectoryPipeline | None = None) -> list[str]:
        """Incrementally ingest one file (or dir) into this space; return the changed doc ids.

        Resolves ``path`` relative to ``manifest.source_root`` (a path outside that root raises
        :class:`~indx.errors.ArchiveError`). Builds (or accepts) a :class:`DirectoryPipeline`
        whose components mirror ``manifest.components`` (offline core when the manifest is empty),
        rebuilds a ``jsonl`` store from ``self.chunks`` (mirroring :meth:`search`), runs the shared
        :func:`indx.pipeline.pipeline.ingest_path` helper restricted to ``path``. New
        ``Document``/``Chunk``/``Relation`` rows are appended, replacing any prior doc with the same
        stable id (idempotent re-add). Relations stay consistent (dangling edges dropped),
        ``chunk.source`` is re-stamped, and vectors are upserted into the rebuilt store. The lists
        are re-canonicalized so an equivalent mutation sequence reseals byte-identically.
        """
        from indx.pipeline.pipeline import ingest_path

        root, rel_paths = self._resolve_add_target(path)
        pipe = pipeline if pipeline is not None else self._pipeline_for_manifest()
        store = self._live_store()
        result = ingest_path(pipe, root, rel_paths, store=store)

        new_ids = [d.id for d in result.documents]
        # Replace any prior doc with the same stable id (idempotent re-add): drop its rows first.
        for doc_id in new_ids:
            self._drop_doc_rows(doc_id)
        self.documents_.extend(result.documents)
        self.chunks.extend(result.chunks)

        if self.manifest.embedding_model is None and result.embedding_model is not None:
            self.manifest.embedding_model = result.embedding_model
            self.manifest.embedding_dim = result.embedding_dim

        self._rederive_relations(result)
        self._restamp_sources()
        self._prune_relations()
        self._canonicalize()
        return new_ids

    def update(self, path: str | Path, *, pipeline: DirectoryPipeline | None = None) -> list[str]:
        """Re-ingest a changed file: ``remove`` its old rows/vectors, then ``add`` (Feature 3).

        Returns the changed doc ids. A no-op-content re-ingest still round-trips deterministically
        because the stable doc/chunk ids are content-position addressed.
        """
        _, rel_paths = self._resolve_add_target(path)
        for rel in rel_paths or []:
            self.remove(rel)
        return self.add(path, pipeline=pipeline)

    def remove(self, target: str | Path) -> list[str]:
        """Drop a document (by ``doc_id`` or path), its chunks, relations, and vectors (Feature 3).

        Returns the removed doc ids (``[]`` if nothing matched — a clean, idempotent no-op, not an
        error, so ``indx rm`` is scriptable). Drops every relation naming the removed doc id in
        either direction, then prunes any now-dangling chunk-level edge, and ``store.delete``s the
        removed chunk ids from a rebuilt store (vectors are gone on the next reseal/search).
        """
        doc_id = self._resolve_doc_id(target)
        if doc_id is None or self.document(doc_id) is None:
            return []
        removed_chunk_ids = [c.id for c in self.chunks_for(doc_id)]
        # Build the live store from the current chunks BEFORE dropping the rows, otherwise the
        # rebuilt store would not contain ``removed_chunk_ids`` and ``delete`` would be a no-op.
        store = self._live_store() if removed_chunk_ids else None
        self._drop_doc_rows(doc_id)
        self.relations = [r for r in self.relations if r.src != doc_id and r.dst != doc_id]
        self._prune_relations()
        if store is not None:
            store.delete(removed_chunk_ids)
        self._canonicalize()
        return [doc_id]

    # -------------------------------------------------------------- CRUD helpers

    def _live_store(self) -> VectorStore:
        """Rebuild a ``jsonl`` store from ``self.chunks`` (mirror :meth:`search`).

        An archive-loaded space has no live pipeline store, so every mutator rebuilds one from the
        space's own chunks to upsert/delete against (zero-dep offline CRUD).
        """
        from typing import cast

        from indx.registry import get_store

        store = cast("VectorStore", get_store("jsonl"))
        store.upsert(self.chunks)
        return store

    def _pipeline_for_manifest(self) -> DirectoryPipeline:
        """Construct a :class:`DirectoryPipeline` mirroring ``manifest.components`` (Feature 3).

        Each slot falls back to the offline core when the manifest does not name it
        (``parser=plaintext``, ``llm=none``, ``embedder=hash``, ``store=jsonl``), so archive-loaded
        CRUD works with zero deps and never reaches for a cloud backend.
        """
        from indx.pipeline.pipeline import DirectoryPipeline

        comp = self.manifest.components
        return DirectoryPipeline(
            parser=comp.get("parser") or "plaintext",
            llm=comp.get("llm") or "none",
            vlm=comp.get("vlm") or "none",
            embedder=comp.get("embedder") or "hash",
            store="jsonl",
            config=None,
        )

    def _resolve_add_target(self, path: str | Path) -> tuple[Path, list[str]]:
        """Resolve an ``add``/``update`` target to ``(walk_root, rel_paths_under_root)``.

        ``manifest.source_root`` is the indexed root; the ingest walks that root but restricts the
        produced documents to ``path``'s root-relative form so ``Document.path`` stays portable. A
        path outside the source root raises :class:`~indx.errors.ArchiveError`.
        """
        from indx.errors import ArchiveError
        from indx.utils.io import iter_files

        root = Path(self.manifest.source_root)
        if not root.is_dir():
            raise ArchiveError(
                f"cannot ingest: source root {self.manifest.source_root!r} is not a directory "
                "(CRUD ingests files under the space's original indexed root)"
            )
        target = Path(path)
        abs_target = target if target.is_absolute() else (root / target)
        try:
            rel_root = abs_target.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ArchiveError(
                f"cannot ingest {path!r}: it is not under the space's source root "
                f"{self.manifest.source_root!r}"
            ) from exc

        abs_resolved = abs_target.resolve()
        if abs_resolved.is_dir():
            rels = [f.relative_to(root.resolve()).as_posix() for f in iter_files(abs_resolved)]
        else:
            rels = [rel_root.as_posix()]
        return root, rels

    def _resolve_doc_id(self, target: str | Path) -> str | None:
        """Map a CLI/SDK ``target`` to a doc id (Feature 3).

        An exact id match wins; else a match on a document's relative ``path``; else
        ``stable_hash(rel_path)`` for a path that is not (yet) in the space — so a caller can
        ``remove`` by the same path string they would ``add``.
        """
        from indx.utils.hashing import stable_hash

        s = str(target)
        if any(d.id == s for d in self.documents_):
            return s
        # Normalize a path target to its root-relative posix form for matching.
        norm = Path(s).as_posix()
        for d in self.documents_:
            if d.path == s or d.path == norm:
                return d.id
        root = Path(self.manifest.source_root)
        if root.is_dir():
            abs_target = Path(s) if Path(s).is_absolute() else (root / s)
            try:
                rel = abs_target.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                rel = norm
            for d in self.documents_:
                if d.path == rel:
                    return d.id
            return stable_hash(rel)
        return stable_hash(norm)

    def _drop_doc_rows(self, doc_id: str) -> None:
        """Drop a document and its chunks (relations handled separately by the caller)."""
        self.documents_ = [d for d in self.documents_ if d.id != doc_id]
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

    def _rederive_relations(self, result: IngestResult) -> None:
        """Recompute relations after an ``add`` so edges touching the new doc(s) are correct.

        Structural edges (``sibling``/``parent``/``continues``) need only ``Document`` metadata, so
        they are recomputed across the WHOLE merged document set (deterministic, folder-local).
        Content edges (``references``/``duplicate-of``) need parsed text, which is only available
        for the freshly-ingested doc(s); those are taken from ``result.relations``. Pre-existing
        content edges between surviving old docs are preserved from the prior relation list. The
        union is deduped; dangling edges are pruned by the caller.
        """
        from indx.core.relation import RelationType

        live_ids = {d.id for d in self.documents_}
        new_ids = {d.id for d in result.documents}
        content = {RelationType.REFERENCES, RelationType.DUPLICATE_OF}

        # Keep prior CONTENT edges only between surviving docs that are NOT being re-derived now.
        preserved = [
            r
            for r in self.relations
            if r.type in content
            and r.src in live_ids
            and r.dst in live_ids
            and r.src not in new_ids
            and r.dst not in new_ids
        ]
        # The ingest's own relations carry the new doc's structural + content edges within the
        # walked subset; recompute structural edges over the full merged document set so the new
        # doc's siblings/parents to existing docs appear.
        structural = self._structural_relations()
        new_content = [r for r in result.relations if r.type in content]

        merged: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        for r in (*structural, *new_content, *preserved):
            key = (str(r.type), r.src, r.dst)
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        self.relations = merged

    def _structural_relations(self) -> list[Relation]:
        """Recompute folder-structural edges (sibling/parent/continues) over all live docs.

        Mirrors :class:`~indx.pipeline.stages.relate.RelateStage` for the metadata-only edges; runs
        it over a context holding every live document with no parsed text, so only structural edges
        are produced (content edges need ``ctx.parsed``).
        """
        from indx.core.context import SpaceContext
        from indx.pipeline.stages.relate import RelateStage

        ctx = SpaceContext(root=Path("."))
        # Sort by path so the per-folder grouping (dict insertion order) is deterministic and
        # matches a fresh full build regardless of the order rows were appended by CRUD.
        ctx.space.documents_ = sorted(self.documents_, key=lambda d: d.path)
        RelateStage().run(ctx)
        return list(ctx.space.relations)

    def _prune_relations(self) -> None:
        """Drop any Relation whose src/dst no longer names a live document or chunk id."""
        live = {d.id for d in self.documents_} | {c.id for c in self.chunks}
        self.relations = [r for r in self.relations if r.src in live and r.dst in live]

    def _restamp_sources(self) -> None:
        """Re-stamp ``Source(path, folder, type)`` onto every chunk by ``doc_id``.

        The exact logic of :func:`indx.pipeline.stages.pack.stamp_sources`, applied to ``self`` so
        ``hit.source.path`` keeps working after a mutation.
        """
        from indx.core.source import Source

        by_doc = {d.id: d for d in self.documents_}
        for chunk in self.chunks:
            doc = by_doc.get(chunk.doc_id)
            if doc is not None:
                chunk.source = Source(path=doc.path, folder=doc.folder, type=doc.doc_type)

    def _canonicalize(self) -> None:
        """Sort the row lists into a canonical order so equivalent mutations reseal identically.

        ``documents_`` by ``path``; ``chunks`` by ``(owning document's path, position)`` so chunks
        follow their document in the same path order a fresh full build of a flat directory emits.
        ``relations`` are NOT re-sorted: structural edges are produced in
        :class:`~indx.pipeline.stages.relate.RelateStage` emission order (folder-group, path-sorted)
        which already matches a fresh build, so preserving that order is what makes "build then add"
        seal identically to "build with everything from the start" (for content with no
        cross-document references). The writer emits members in list order.
        """
        self.documents_.sort(key=lambda d: d.path)
        path_by_doc = {d.id: d.path for d in self.documents_}
        self.chunks.sort(key=lambda c: (path_by_doc.get(c.doc_id, ""), c.position))

    # ------------------------------------------------------------------ persistence

    @classmethod
    def load(cls, archive: str, *, members: Iterable[str] | None = None) -> KnowledgeSpace:
        """Open a sealed ``.indx`` archive into a :class:`KnowledgeSpace` (sdk.md §load).

        ``members=None`` reads the whole archive (unchanged). Passing a subset of
        ``{"documents", "chunks", "relations"}`` reads/validates only those members (Feature 1,
        selective archive load); the others come back empty and are never decompressed. An
        unknown member name raises :class:`~indx.errors.ArchiveError` (exit 4).
        """
        from pathlib import Path

        from indx.archive import read_archive

        return read_archive(Path(archive), members=members)

    @classmethod
    def load_part(cls, archive: str, member: str) -> list[Document] | list[Chunk] | list[Relation]:
        """Load a single member of a sealed archive (Feature 1, selective archive load).

        ``member`` ∈ ``{"documents", "chunks", "relations"}`` → the corresponding model list.
        Reads/validates only that member (plus the always-read manifest guard and its checksum
        leg). An unknown member (including ``"manifest"``, which is not a loadable content
        member) raises :class:`~indx.errors.ArchiveError` (exit 4).
        """
        space = cls.load(archive, members=(member,))
        parts: dict[str, list[Document] | list[Chunk] | list[Relation]] = {
            "documents": space.documents_,
            "chunks": space.chunks,
            "relations": space.relations,
        }
        return parts[member]

    def save(self, archive: str) -> None:
        """Seal this space into a portable ``.indx`` archive (sdk.md §save)."""
        from pathlib import Path

        from indx.archive import write_archive

        write_archive(self, Path(archive))
