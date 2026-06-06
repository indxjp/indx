"""DirectoryPipeline — build stages from slot selections and run them in order.

This is the SDK surface that mirrors the CLI (FR-SDK-1): construct with component names,
``run()`` on a directory, get a :class:`KnowledgeSpace` back. Defaults are the documented
zero-config stack (technical-spec §8.2); selecting an extra-gated default without its extra
installed raises a friendly :class:`~indx.errors.MissingExtraError` at construction.

Cross-cutting run behavior (strict / resume / dry-run / jobs) lives **here**, at the
orchestration layer, rather than inside the individual stages, so the stage files stay focused
on their single phase and remain swappable:

* ``strict`` — after every stage, any ``kind="skip"`` error gained on the context is promoted
  to a fatal :class:`~indx.errors.StageError` (technical-spec §3.4).
* ``resume`` — Parse and Embed outputs are reused from a content-addressed cache under
  ``<out>/.indx-cache/`` when their inputs and components are unchanged (technical-spec §10.3).
* ``jobs`` — the worker count threaded into Parse fan-out; output stays byte-identical
  regardless of worker count because results are re-sorted before id assignment (§10.4).
* ``plan`` / ``dry_run`` — run Walk only and return the plan (files, folders, components)
  without invoking any model/embed work (FR-CLI-6).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, cast

from indx.config import Config, load_config
from indx.config.defaults import (
    DEFAULT_EMBEDDER,
    DEFAULT_FORMAT,
    DEFAULT_LLM,
    DEFAULT_PARSER,
    DEFAULT_STORE,
    DEFAULT_VLM,
)
from indx.core.context import SpaceContext, StageErrorRecord
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.core.parsed import ParsedDoc
from indx.embed.base import Embedder
from indx.errors import RegistryError, StageError
from indx.pipeline.stage import Stage
from indx.pipeline.stages import (
    ChunkStage,
    EnrichStage,
    PackStage,
    RelateStage,
    WalkStage,
)
from indx.registry import (
    discover_stages,
    get_embedder,
    get_parser,
    get_store,
    get_writer,
)
from indx.store.base import VectorStore
from indx.utils.cache import StageCache, sha256_file, sha256_text

# Documented zero-config defaults per slot (sdk.md "Constructor"). Used when a slot is left
# ``None`` and ``indx.toml`` does not select it. The values come from the single source of
# truth in ``indx.config.defaults`` so the SDK and the pydantic schema can never drift.
_DEFAULTS: Final[dict[str, str]] = {
    "parser": DEFAULT_PARSER,
    "llm": DEFAULT_LLM,
    "vlm": DEFAULT_VLM,
    "embedder": DEFAULT_EMBEDDER,
    "store": DEFAULT_STORE,
    "output": DEFAULT_FORMAT,
}

# Stores that pre-size a fixed-width vector column/collection at construction and accept a
# ``dim`` kwarg. For these, the pipeline must forward the embedder's *actual* dimensionality
# so a 1536-dim (openai) or 1024-dim (bge/cohere) vector is not upserted into a default-768
# collection (which fails on upsert). Stores absent here (``jsonl``/``chroma``) size lazily
# from the first vector and ignore/reject a ``dim`` kwarg, so it is never forwarded to them.
_DIM_SIZED_STORES: Final[frozenset[str]] = frozenset(
    {
        "qdrant",
        "pgvector",
        "lancedb",
        "s3vectors",
        "opensearch",
        "azure-search",
        "bigquery",
        "vertex-vector",
    }
)

# Map a documented output format name to its writer slot name in the registry
# (mirrors cli._render._WRITER_NAMES so the SDK and CLI resolve writers identically).
_WRITER_NAMES: Final[dict[str, str]] = {
    ".indx": "indx",
    "indx": "indx",
    "jsonl": "jsonl",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
}

# Sentinel so ``run(src)`` (out omitted -> in-memory) is distinguishable from
# ``run(src, out=None)`` (out explicitly disabled). Both stay in memory; the sentinel keeps
# the CLI's ``run(dir, on_stage=...)`` path (which writes separately) from double-writing.
_UNSET: Final = object()


class BuildPlan:
    """The result of a dry-run (FR-CLI-6): what *would* be processed, without doing it.

    Attributes:
        root: The directory that would be indexed.
        documents: The :class:`~indx.core.document.Document` records produced by Walk.
        components: The resolved slot → name mapping the build would use.
        embed: Whether stage 06 (embed) would run.
        enrich: Whether stage 05 (enrich) would run.
    """

    def __init__(
        self,
        *,
        root: Path,
        documents: list[Document],
        components: dict[str, str],
        embed: bool,
        enrich: bool,
    ) -> None:
        self.root = root
        self.documents = documents
        self.components = components
        self.embed = embed
        self.enrich = enrich

    @property
    def folders(self) -> list[str]:
        """Distinct folders (root-relative) that would be processed, sorted."""
        seen = {"/".join(d.lineage) for d in self.documents}
        return sorted(seen)


def _slot_name(
    value: object | str | None,
    *,
    slot: str,
    config_value: str | None,
) -> tuple[str | None, object | None]:
    """Resolve one constructor slot to ``(name, instance)``.

    Resolution order (sdk.md "Constructor"): explicit instance/name argument → ``indx.toml``
    → documented default. An instance is used directly (its ``name`` attribute, if any, is
    recorded for provenance); a string is resolved later via the registry; ``None`` falls back
    to config then the default.
    """
    if value is None:
        name = config_value if config_value is not None else _DEFAULTS[slot]
        return name, None
    if isinstance(value, str):
        return value, None
    # A concrete instance was passed: use it directly, recording its name for provenance.
    return cast("str | None", getattr(value, "name", None)), value


def _build_store(
    name: str,
    *,
    embedder: Embedder | None,
    options: dict[str, Any] | None = None,
) -> Any:
    """Resolve and construct a store by name, forwarding the embedder's real ``dim``.

    Pre-sizing stores (:data:`_DIM_SIZED_STORES`) create a fixed-width collection/column at
    construction, so they must be told the *actual* embedding width up front — otherwise a
    1536-dim (openai) vector is upserted into the default-768 collection and the write fails.
    When an embedder is present and the user did not already pin ``dim`` via the store's
    config sub-table, the embedder's ``dim`` is injected into the constructor kwargs. An
    explicit ``[store.<backend>] dim`` config value always wins. Stores that size lazily
    (``jsonl``/``chroma``) never receive ``dim``.
    """
    kwargs: dict[str, Any] = dict(options or {})
    if (
        embedder is not None
        and name in _DIM_SIZED_STORES
        and "dim" not in kwargs  # an explicit config dim override wins
    ):
        kwargs["dim"] = embedder.dim
    return get_store(name, **kwargs)


class DirectoryPipeline:
    """Ordered, replaceable stages over a shared SpaceContext (FR-SDK-1).

    Each component slot accepts an **instance**, a **name string**, or ``None``. An unset slot
    falls back to ``indx.toml`` (when ``config`` is given or ``./indx.toml`` exists) and then to
    the documented default (sdk.md "Constructor"). Unknown names raise before any stage runs.
    """

    def __init__(
        self,
        *,
        parser: object | str | None = None,
        llm: object | str | None = None,
        vlm: object | str | None = None,
        embedder: object | str | None = None,
        store: object | str | None = None,
        output: object | str | None = None,
        config: str | Config | None = None,
        seed: int = 0,
        enrich: bool = True,
        embed: bool = True,
        strict: bool = False,
        resume: bool = False,
        jobs: int | None = None,
        out: str | Path | None = None,
    ) -> None:
        cfg = config if isinstance(config, Config) else load_config(config)
        slot_opts = cfg.slot_options()

        # Resolve each slot to (name, optional pre-built instance). Names are recorded in the
        # manifest for provenance and used to resolve registry-backed slots lazily.
        p_name, p_inst = _slot_name(parser, slot="parser", config_value=cfg.parser.engine)
        l_name, _ = _slot_name(llm, slot="llm", config_value=cfg.enrich.llm)
        v_name, _ = _slot_name(vlm, slot="vlm", config_value=cfg.enrich.vlm)
        e_name, e_inst = _slot_name(embedder, slot="embedder", config_value=cfg.embed.model)
        s_name, s_inst = _slot_name(store, slot="store", config_value=cfg.store.backend)
        o_name, o_inst = _slot_name(output, slot="output", config_value=cfg.output.format)

        self._names = {
            "parser": p_name or _DEFAULTS["parser"],
            "llm": l_name or _DEFAULTS["llm"],
            "vlm": v_name or _DEFAULTS["vlm"],
            "embedder": e_name or _DEFAULTS["embedder"],
            "store": s_name or _DEFAULTS["store"],
        }
        self.seed = seed
        self.strict = strict
        self.resume = resume
        # Default worker count is the CPU count (technical-spec §10.1). Always >= 1.
        self.jobs = jobs if jobs and jobs > 0 else (os.cpu_count() or 1)
        self.out = Path(out) if out is not None else None
        self._enrich = enrich
        self._embed = embed

        # Build (or accept) each component. A passed instance is used verbatim; a name is
        # resolved via the registry, forwarding the per-slot config sub-table as kwargs.
        self._parser = (
            p_inst
            if p_inst is not None
            else get_parser(self._names["parser"], **slot_opts.get("parser", {}))
        )
        self._embedder: Embedder | None = (
            (
                cast("Embedder", e_inst)
                if e_inst is not None
                else get_embedder(self._names["embedder"], **slot_opts.get("embedder", {}))
            )
            if embed
            else None
        )
        # Build the store after the embedder so a pre-sizing store (qdrant/pgvector/lancedb)
        # is sized to the embedder's real dim, not the hard-coded default (T10 fix).
        self._store = (
            s_inst
            if s_inst is not None
            else _build_store(
                self._names["store"],
                embedder=self._embedder,
                options=slot_opts.get("store", {}),
            )
        )
        # Track whether the store is a user-supplied instance vs registry-built. A registry-built
        # dim-sized store is sized to the *current* embedder; rebinding only the embedder via
        # ``use()`` must re-size it (T10), but a user-supplied store is never silently rebuilt.
        self._store_user_supplied = s_inst is not None
        self._output_name = o_name or _DEFAULTS["output"]
        self._output = o_inst  # writer instance; resolved lazily by name when None
        self._output_opts = slot_opts.get("output", {})

        self._cache = StageCache(self.out, enabled=resume)
        self._stages: list[Stage] = [
            WalkStage(),
            cast(
                "Stage",
                ResumableParseStage(self._parser, self._names["parser"], self._cache, self.jobs),
            ),
            ChunkStage(),
            RelateStage(),
            *([EnrichStage()] if enrich else []),
            *(
                [
                    cast(
                        "Stage",
                        ResumablePackStage(
                            self._embedder,
                            cast("VectorStore", self._store),
                            self._names["embedder"],
                            self._cache,
                        ),
                    )
                ]
                if embed and self._embedder is not None
                else []
            ),
        ]
        # Merge third-party stages advertised under ``[project.entry-points.'indx.stages']``
        # (file-architecture §6); appended by default so built-ins keep canonical order.
        self._merge_plugin_stages()

    def _merge_plugin_stages(self) -> None:
        """Append any discovered third-party stages whose name is not already present."""
        existing = {s.name for s in self._stages}
        for name, factory in discover_stages().items():
            if name in existing:
                continue
            stage = factory() if callable(factory) else factory
            self._stages.append(cast("Stage", stage))
            existing.add(name)

    @property
    def store(self) -> VectorStore:  # exposed so the CLI/SDK can query right after a build
        return cast(VectorStore, self._store)

    @property
    def embedder(self) -> Embedder:
        return cast(Embedder, self._embedder)

    def _new_context(self, directory: str | Path) -> SpaceContext:
        from indx.utils.zip_input import extract_zip, is_zip_input

        given = Path(directory)
        if is_zip_input(given):
            # ZIP build input: extract into a temp dir that becomes the walk root, but keep
            # the manifest's source_root pointing at the original archive so it stays portable
            # and self-describing. The temp dir is cleaned up by run()/plan().
            root = extract_zip(given)
            tmp_root: Path | None = root
            source_root = str(given)
        elif given.is_dir():
            root, tmp_root, source_root = given, None, str(given)
        else:
            raise NotADirectoryError(f"not a directory or .zip: {given}")
        ctx = SpaceContext(root=root, tmp_root=tmp_root, seed=self.seed)
        ctx.space.manifest = Manifest(source_root=source_root, components=dict(self._names))
        return ctx

    # ------------------------------------------------------------------ stage management
    def stages(self) -> list[Stage]:
        """The current ordered stage list (sdk.md "Stage management").

        Returns the live list so ``pipeline.stages().insert(...)`` / indexing mutate the
        pipeline in place, matching the fluent helpers below.
        """
        return self._stages

    def insert(self, index: int, stage: Stage) -> DirectoryPipeline:
        """Insert a custom stage at a 0-based position; returns ``self`` for chaining."""
        self._stages.insert(index, stage)
        return self

    def append(self, stage: Stage) -> DirectoryPipeline:
        """Append a stage to the end; returns ``self`` for chaining."""
        self._stages.append(stage)
        return self

    def replace(self, name: str, stage: Stage) -> DirectoryPipeline:
        """Replace the stage with the given ``name``; returns ``self`` for chaining."""
        for i, existing in enumerate(self._stages):
            if existing.name == name:
                self._stages[i] = stage
                return self
        raise RegistryError(f"no stage named '{name}' to replace")

    def drop(self, name: str) -> DirectoryPipeline:
        """Remove the named stage (e.g. ``"enrich"`` / ``"embed-pack"``); returns ``self``."""
        kept = [s for s in self._stages if s.name != name]
        if len(kept) == len(self._stages):
            raise RegistryError(f"no stage named '{name}' to drop")
        self._stages = kept
        return self

    # ------------------------------------------------------------------ component binding
    def use(self, **components: object | str) -> DirectoryPipeline:
        """Swap components by keyword after construction; returns ``self`` for chaining.

        Accepts ``parser=``/``llm=``/``vlm=``/``embedder=``/``store=``/``output=`` as either an
        instance or a name string (sdk.md "Component binding"). Rebinding ``embedder``/``store``
        rebuilds the Embed+Pack stage so the new component takes effect on the next ``run``.
        """
        valid = {"parser", "llm", "vlm", "embedder", "store", "output"}
        unknown = set(components) - valid
        if unknown:
            raise RegistryError(f"unknown component slot(s): {sorted(unknown)}")

        # Defer the store build until after the loop so it sees the possibly-rebound embedder
        # (a pre-sizing store must be sized to the new embedder's real dim, T10).
        store_pending: tuple[str, object | None] | None = None

        for slot, value in components.items():
            name, inst = _slot_name(value, slot=slot, config_value=None)
            resolved = name or _DEFAULTS[slot]
            if slot in self._names:
                self._names[slot] = resolved
            if slot == "parser":
                self._parser = inst if inst is not None else get_parser(resolved)
            elif slot == "embedder":
                self._embedder = (
                    cast(
                        "Embedder",
                        inst if inst is not None else get_embedder(resolved),
                    )
                    if self._embed
                    else None
                )
            elif slot == "store":
                store_pending = (resolved, inst)
            elif slot == "output":
                self._output_name = resolved
                self._output = inst
            # llm / vlm are recorded for provenance; the deterministic Enrich stage uses no model.

        if store_pending is not None:
            resolved, inst = store_pending
            self._store = (
                inst if inst is not None else _build_store(resolved, embedder=self._embedder)
            )
            self._store_user_supplied = inst is not None
        elif (
            "embedder" in components
            and self._embed
            and not self._store_user_supplied
            and self._names["store"] in _DIM_SIZED_STORES
        ):
            # The dim-sized store was sized to the previous embedder; re-size it to the new one
            # so the next ``run()`` does not upsert a mismatched-width vector (T10).
            self._store = _build_store(self._names["store"], embedder=self._embedder)

        self._rebind_stages()
        return self

    def _rebind_stages(self) -> None:
        """Re-point the built-in parse/embed-pack stages at freshly bound components.

        Custom/plugin stages and their positions are preserved; only the registry-backed
        built-ins are swapped so ``use()`` after ``insert()``/``drop()`` stays consistent.
        """
        new: list[Stage] = []
        for stage in self._stages:
            if stage.name == "parse" and isinstance(stage, ResumableParseStage):
                new.append(
                    ResumableParseStage(self._parser, self._names["parser"], self._cache, self.jobs)
                )
            elif stage.name == "embed-pack" and isinstance(stage, ResumablePackStage):
                if self._embed and self._embedder is not None:
                    new.append(
                        ResumablePackStage(
                            self._embedder,
                            cast("VectorStore", self._store),
                            self._names["embedder"],
                            self._cache,
                        )
                    )
                # else: embed disabled now — drop the stage
            else:
                new.append(stage)
        self._stages = new

    def _writer(self) -> Any:
        """The chosen output writer instance (resolve by name when an instance wasn't given)."""
        if self._output is not None:
            return self._output
        slot = _WRITER_NAMES.get(self._output_name, self._output_name)
        return get_writer(slot, **self._output_opts)

    def plan(self, directory: str | Path) -> BuildPlan:
        """Run Walk only and return the build plan — no parse/enrich/embed (FR-CLI-6).

        This is the SDK surface behind ``--dry-run``: it tells the user which files and folders
        would be processed and which components are selected, without touching any model.

        Args:
            directory: The directory (or extracted ZIP) that would be indexed.

        Returns:
            A :class:`BuildPlan` describing the would-be run.

        Raises:
            NotADirectoryError: If ``directory`` is neither a directory nor a ``.zip``.
        """
        from indx.utils.zip_input import cleanup_extracted

        ctx = self._new_context(directory)
        try:
            WalkStage().run(ctx)
            # Report the original input path (the archive, for a ZIP), not the temp extraction dir.
            return BuildPlan(
                root=Path(ctx.space.manifest.source_root),
                documents=list(ctx.space.documents_),
                components=dict(self._names),
                embed=self._embed,
                enrich=self._enrich,
            )
        finally:
            cleanup_extracted(ctx.tmp_root)

    # ``dry_run`` is a readable alias for ``plan`` so both vocabularies work from the SDK.
    dry_run = plan

    def run(
        self,
        directory: str | Path,
        out: str | Path | None = _UNSET,  # type: ignore[assignment]
        *,
        on_stage: Callable[[str], None] | None = None,
        name: str = "handbook",
    ) -> KnowledgeSpace:
        """Run every registered stage over ``directory``; return the :class:`KnowledgeSpace`.

        When ``out`` is given (sdk.md "Execution"), the output layout is sealed to that path via
        the chosen output writer; when ``out`` is ``None`` (or omitted) the space stays in
        memory. ``out`` omitted is distinct from ``out=None`` only so the CLI's
        ``run(dir, on_stage=...)`` path — which writes separately — is not double-written.
        """
        from indx.utils.zip_input import cleanup_extracted

        ctx = self._new_context(directory)
        try:
            for stage in self._stages:
                if on_stage is not None:
                    on_stage(stage.name)
                ctx = stage.run(ctx)
                if self.strict:
                    _enforce_strict(stage.name, ctx)
            space = ctx.space

            if out is not _UNSET and out is not None:
                self._writer().write(space, Path(out), name=name)
            return space
        finally:
            cleanup_extracted(ctx.tmp_root)


def _enforce_strict(stage_name: str, ctx: SpaceContext) -> None:
    """Promote any per-item ``skip`` recorded by the just-run stage to a fatal error (§3.4)."""
    skip = next((e for e in ctx.errors if e.kind == "skip"), None)
    if skip is not None:
        raise StageError(
            skip.stage or stage_name,
            f"--strict: per-item skip promoted to fatal: {skip.message}",
            path=skip.item,
        )


def _sorted_documents(ctx: SpaceContext) -> list[Document]:
    """Documents in deterministic order so concurrency never affects output (§10.4)."""
    return sorted(ctx.space.documents_, key=lambda d: d.path)


class ResumableParseStage:
    """Parse fan-out with ``--jobs`` workers and a content-addressed ``--resume`` cache.

    Delegates the actual parsing to the configured :class:`~indx.parsers.base.Parser`, but owns
    the cross-cutting concerns the bare :class:`ParseStage` does not: a bounded thread pool
    (technical-spec §10.1) and reuse of unchanged outputs from the stage cache (§10.3). Results
    are merged into ``ctx.parsed`` in deterministic document order so the worker count never
    changes the output (§10.4). A per-file parse failure is recorded as a ``kind="skip"`` error
    on the context (which ``--strict`` then promotes to fatal).
    """

    name = "parse"

    def __init__(
        self,
        parser: object,
        component_id: str,
        cache: StageCache,
        jobs: int,
    ) -> None:
        # ``parser`` satisfies the Parser protocol; typed loosely because the registry returns
        # ``Any``. Only ``.parse(path)`` is used.
        self._parser = parser
        self._component_id = component_id
        self._cache = cache
        self._jobs = max(1, jobs)

    def run(self, ctx: SpaceContext) -> SpaceContext:
        docs = _sorted_documents(ctx)
        results: dict[str, ParsedDoc | StageErrorRecord] = {}

        def work(doc: Document) -> tuple[str, ParsedDoc | StageErrorRecord]:
            path = ctx.root / doc.path
            try:
                input_hash = sha256_file(path)
                cached = self._cache.get(self.name, input_hash, self._component_id)
                if cached is not None:
                    return doc.id, ParsedDoc.model_validate(cached)
                parsed = self._parser.parse(path)  # type: ignore[attr-defined]
                self._cache.put(self.name, input_hash, self._component_id, parsed.model_dump())
            except Exception as exc:  # per-file failure → skip (technical-spec §3.4)
                return doc.id, StageErrorRecord(
                    stage=self.name, kind="skip", item=doc.path, message=str(exc)
                )
            return doc.id, parsed

        # A single worker stays fully sequential (cheap, and what the tests assert by default).
        if self._jobs == 1 or len(docs) <= 1:
            for doc in docs:
                doc_id, res = work(doc)
                results[doc_id] = res
        else:
            with ThreadPoolExecutor(max_workers=self._jobs) as pool:
                for doc_id, res in pool.map(work, docs):
                    results[doc_id] = res

        # Re-merge in deterministic order regardless of completion order (§10.4).
        for doc in docs:
            res = results[doc.id]
            if isinstance(res, StageErrorRecord):
                ctx.errors.append(res)
            else:
                ctx.parsed[doc.id] = res
        return ctx


class ResumablePackStage:
    """Embed + pack with a content-addressed ``--resume`` cache for per-chunk vectors.

    Delegates to a real :class:`PackStage` when caching is off (the common path). With
    ``--resume`` on, vectors whose ``(text, embedder)`` is unchanged are reused from the cache,
    and only the cache-missing chunks are sent to ``Embedder.embed`` — then upserted and packed
    exactly as the underlying stage would (technical-spec §10.3). Output is identical regardless
    of cache hits.
    """

    name = "embed-pack"

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        component_id: str,
        cache: StageCache,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._component_id = component_id
        self._cache = cache
        self._inner = PackStage(embedder, store)

    def run(self, ctx: SpaceContext) -> SpaceContext:
        if not self._cache.enabled:
            return self._inner.run(ctx)

        chunks = ctx.space.chunks
        if chunks:
            hashes = [sha256_text(c.text) for c in chunks]
            vectors: list[list[float] | None] = [
                self._cache.get(self.name, h, self._component_id) for h in hashes
            ]
            misses = [i for i, v in enumerate(vectors) if v is None]
            if misses:
                fresh = self._embedder.embed([chunks[i].text for i in misses])
                for i, vec in zip(misses, fresh, strict=True):
                    vectors[i] = vec
                    self._cache.put(self.name, hashes[i], self._component_id, vec)
            for chunk, resolved in zip(chunks, vectors, strict=True):
                assert resolved is not None  # every slot filled (hit or fresh) above
                chunk.embedding = resolved
            self._store.upsert(chunks)
        ctx.space.manifest.embedding_model = self._embedder.name
        ctx.space.manifest.embedding_dim = self._embedder.dim
        return ctx
