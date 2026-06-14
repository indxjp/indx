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

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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
from indx.core.chunk import Chunk
from indx.core.context import SpaceContext, StageErrorRecord
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.core.parsed import ParsedDoc
from indx.core.relation import Relation
from indx.embed.base import Embedder
from indx.errors import RegistryError, StageError
from indx.pipeline.filters import WalkFilter
from indx.pipeline.stage import Stage
from indx.pipeline.stages import (
    ChunkStage,
    EnrichStage,
    PackStage,
    RelateStage,
    WalkStage,
)
from indx.pipeline.stages.pack import stamp_sources
from indx.registry import (
    discover_stages,
    get_embedder,
    get_parser,
    get_store,
    get_writer,
)
from indx.store.base import VectorStore, close_store
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
        skipped: Files dropped by the configured WalkFilter (Feature 5); 0 when unfiltered.
    """

    def __init__(
        self,
        *,
        root: Path,
        documents: list[Document],
        components: dict[str, str],
        embed: bool,
        enrich: bool,
        skipped: int = 0,
    ) -> None:
        self.root = root
        self.documents = documents
        self.components = components
        self.embed = embed
        self.enrich = enrich
        self.skipped = skipped

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
        filter: WalkFilter | None = None,
    ) -> None:
        cfg = config if isinstance(config, Config) else load_config(config)
        slot_opts = cfg.slot_options()

        # Resolve each slot to (name, optional pre-built instance). Names are recorded in the
        # manifest for provenance and used to resolve registry-backed slots lazily.
        p_name, p_inst = _slot_name(parser, slot="parser", config_value=cfg.parser.engine)
        l_name, l_inst = _slot_name(llm, slot="llm", config_value=cfg.enrich.llm)
        v_name, _ = _slot_name(vlm, slot="vlm", config_value=cfg.enrich.vlm)
        e_name, e_inst = _slot_name(embedder, slot="embedder", config_value=cfg.embed.model)
        s_name, s_inst = _slot_name(store, slot="store", config_value=cfg.store.backend)
        o_name, o_inst = _slot_name(output, slot="output", config_value=cfg.output.format)

        # H2 — build-time LLM enrichment, but ONLY on an EXPLICIT selection. The vlm provenance
        # stays ``none`` (no build-time VLM enricher). The llm is used for enrichment iff the user
        # *explicitly* asked for one: either the ``llm=`` constructor arg was provided (not None),
        # or ``[enrich] llm`` was set to a real value in config. It must NOT be resolved/called on
        # the zero-config / air-gapped default path (``cfg.enrich.llm`` defaults to DEFAULT_LLM),
        # or the offline promise — and the whole test suite — breaks. So we never fall back to the
        # documented default for enrichment: an unset / "" / "none" llm means "no LLM enrichment".
        del v_name  # resolved but unused for build provenance (no build-time VLM enricher)
        self._enrich_llm = self._resolve_enrich_llm(
            arg=llm, l_name=l_name, l_inst=l_inst, cfg_llm=cfg.enrich.llm
        )
        self._names = {
            "parser": p_name or _DEFAULTS["parser"],
            # Honest provenance: the real llm name only when an explicit LLM enriches this build;
            # otherwise ``none`` (the deterministic scaffold ran, claiming an LLM would be a lie).
            "llm": self._enrich_llm[0] if self._enrich_llm is not None else "none",
            "vlm": "none",
            "embedder": e_name or _DEFAULTS["embedder"],
            "store": s_name or _DEFAULTS["store"],
        }
        # The configured ``[enrich] metadata`` list selects which fields the scaffold populates
        # (technical-spec §9); forwarded to the build-time EnrichStage below so a config of e.g.
        # ``metadata = ["summary"]`` no longer silently populates topics/tags/type as well.
        self._enrich_metadata = list(cfg.enrich.metadata)
        self.seed = seed
        self.strict = strict
        self.resume = resume
        # Default worker count is the CPU count (technical-spec §10.1). Always >= 1.
        self.jobs = jobs if jobs and jobs > 0 else (os.cpu_count() or 1)
        self.out = Path(out) if out is not None else None
        self._enrich = enrich
        self._embed = embed
        # Feature 5: the build-time WalkFilter, re-read by both run() (via the WalkStage it
        # constructs) and plan() (which builds its own WalkStage on the same filter).
        self._filter = filter

        # Build (or accept) each component. A passed instance is used verbatim; a name is
        # resolved via the registry, forwarding the per-slot config sub-table as kwargs.
        # Retain the per-slot parser config: it is folded into the resume cache component_id so a
        # parser config change that alters parse output is a clean cache miss (bug S2).
        self._parser_opts = dict(slot_opts.get("parser", {}))
        self._parser = (
            p_inst if p_inst is not None else get_parser(self._names["parser"], **self._parser_opts)
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
        # Retain the per-slot store config so a ``use()`` rebuild (rebinding the embedder, or the
        # store itself by name) re-applies the user's pinned connection settings (qdrant url,
        # pgvector dsn, lancedb uri, explicit dim, …) instead of falling back to defaults.
        self._store_opts = slot_opts.get("store", {})
        self._output_name = o_name or _DEFAULTS["output"]
        self._output = o_inst  # writer instance; resolved lazily by name when None
        self._output_opts = slot_opts.get("output", {})

        self._cache = StageCache(self.out, enabled=resume)
        self._stages: list[Stage] = [
            WalkStage(self._filter),
            cast(
                "Stage",
                ResumableParseStage(
                    self._parser, self._parser_component_id(), self._cache, self.jobs
                ),
            ),
            ChunkStage(),
            RelateStage(),
            *(
                [
                    EnrichStage(
                        metadata=self._enrich_metadata,
                        llm=(self._enrich_llm[1] if self._enrich_llm is not None else None),
                    )
                ]
                if enrich
                else []
            ),
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

    @staticmethod
    def _resolve_enrich_llm(
        *,
        arg: object | str | None,
        l_name: str | None,
        l_inst: object | None,
        cfg_llm: str,
    ) -> tuple[str, object] | None:
        """Resolve the build-time enrichment LLM, or ``None`` when none is explicitly selected.

        An LLM enriches the build ONLY on an explicit selection (H2):

        * the ``llm=`` constructor arg was provided (not ``None``) — an instance is used directly,
          a name string is resolved via the registry; or
        * ``[enrich] llm`` was set in config to a real value (i.e. something other than the
          documented default :data:`DEFAULT_LLM`, ``""`` or ``"none"``).

        We deliberately never fall back to :data:`DEFAULT_LLM` here: the zero-config / air-gapped
        default path must stay fully offline and deterministic. Returns ``(name, instance)`` ready
        to hand to :class:`EnrichStage`.
        """
        # 1. Explicit constructor instance — use it verbatim.
        if arg is not None and not isinstance(arg, str):
            name = l_name or getattr(arg, "name", None) or "llm"
            return name, arg
        # 2. Explicit constructor name string — resolve via the registry.
        if isinstance(arg, str):
            selected = arg.strip()
            if selected and selected.lower() != "none":
                from indx.registry import get_llm

                return selected, get_llm(selected)
            return None
        # 3. No constructor arg: honour an explicit config selection only (not the default).
        configured = (cfg_llm or "").strip()
        if configured and configured.lower() != "none" and configured != DEFAULT_LLM:
            from indx.registry import get_llm

            return configured, get_llm(configured)
        return None

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
    def components(self) -> dict[str, str]:
        """Resolved slot names as sealed into the manifest (bug #5: honest build provenance).

        The CLI build summary reads this rather than the requested config so it can never
        disagree with ``manifest.components`` — e.g. ``llm``/``vlm`` read ``none`` for a build,
        since the build-time enricher is LLM-free regardless of the requested slot.
        """
        return dict(self._names)

    @property
    def store(self) -> VectorStore:  # exposed so the CLI/SDK can query right after a build
        return cast(VectorStore, self._store)

    @property
    def embedder(self) -> Embedder | None:
        # ``None`` when the pipeline was built (or rebound) with ``embed=False`` — the annotation
        # is honest so callers querying right after a build get a typed Optional, not an
        # ``AttributeError`` from a value the static type claimed was always present.
        return self._embedder

    def close(self) -> None:
        """Release the store this pipeline constructed (best-effort, idempotent).

        Registry-built stores are owned by the pipeline, so their backend resources (DB
        connections, on-disk client handles) are released here. A user-supplied store is left
        untouched — its lifetime belongs to the caller. ``close()`` is a no-op for stores
        without resources to free (the default :meth:`VectorStore.close`). Safe to call more
        than once; after ``close()`` the pipeline must not be ``run()`` again.

        Long-lived hosts (the ``indx app`` build worker, CLI ``build``) should call this — or
        use the pipeline as a context manager — so per-build stores do not accumulate handles.
        """
        if self._store_user_supplied:
            return
        store = self._store
        self._store = None
        if store is not None:
            close_store(store)

    def __enter__(self) -> DirectoryPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
            # llm/vlm provenance stays ``none`` on the build path — rebinding them must not
            # re-introduce the false claim that an LLM ran at build time (the deterministic
            # Enrich stage uses no model). The resolved name is otherwise unused here.
            if slot in self._names and slot not in ("llm", "vlm"):
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

        if store_pending is not None:
            resolved, inst = store_pending
            self._store = (
                inst
                if inst is not None
                else _build_store(resolved, embedder=self._embedder, options=self._store_opts)
            )
            self._store_user_supplied = inst is not None
        elif (
            "embedder" in components
            and self._embed
            and not self._store_user_supplied
            and self._names["store"] in _DIM_SIZED_STORES
        ):
            # The dim-sized store was sized to the previous embedder; re-size it to the new one
            # so the next ``run()`` does not upsert a mismatched-width vector (T10). Forward the
            # pinned store config so the rebuild keeps targeting the configured location.
            self._store = _build_store(
                self._names["store"], embedder=self._embedder, options=self._store_opts
            )

        self._rebind_stages()
        return self

    def _parser_component_id(self) -> str:
        """Resume cache ``component_id`` for the parse stage: name **plus** parser version.

        The cache addresses a parsed doc by ``(stage="parse", sha256(file), component_id)``. Using
        the registry name alone meant a parser whose behavior changed but whose name stayed the
        same would silently serve a stale ParsedDoc under ``--resume`` (bug S2). Folding the
        parser ``version`` (and a stable hash of any per-slot parser config that can alter parse
        output) into the id isolates those caches so a version/config bump is a clean cache miss.

        The Parser Protocol declares only ``name``; concrete adapters add ``version``, so we read
        it defensively (default ``"0"``).
        """
        component_id = f"{self._names['parser']}@{getattr(self._parser, 'version', '0')}"
        if self._parser_opts:
            opts_hash = hashlib.sha256(
                json.dumps(self._parser_opts, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
            component_id = f"{component_id}+{opts_hash}"
        return component_id

    def _rebind_stages(self) -> None:
        """Re-point the built-in parse/embed-pack stages at freshly bound components.

        Custom/plugin stages and their positions are preserved; only the registry-backed
        built-ins are swapped so ``use()`` after ``insert()``/``drop()`` stays consistent.
        """
        new: list[Stage] = []
        for stage in self._stages:
            if stage.name == "parse" and isinstance(stage, ResumableParseStage):
                new.append(
                    ResumableParseStage(
                        self._parser, self._parser_component_id(), self._cache, self.jobs
                    )
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
            WalkStage(self._filter).run(ctx)
            # Report the original input path (the archive, for a ZIP), not the temp extraction dir.
            return BuildPlan(
                root=Path(ctx.space.manifest.source_root),
                documents=list(ctx.space.documents_),
                components=dict(self._names),
                embed=self._embed,
                enrich=self._enrich,
                skipped=ctx.space.skipped_files_,
            )
        finally:
            cleanup_extracted(ctx.tmp_root)

    # ``dry_run`` is a readable alias for ``plan`` so both vocabularies work from the SDK.
    dry_run = plan

    def _resolve_stage_subset(self, requested: Sequence[str]) -> list[Stage]:
        """Filter ``self._stages`` to ``requested``, preserving canonical pipeline order.

        Feature 1 (selective stage execution): the subset is resolved against the **live** stage
        list (so plugin/custom stages and ``--no-embed``/``drop`` removals are honored). An unknown
        name raises :class:`~indx.errors.RegistryError` (exit 3) naming the offender and the live
        stages — matching the existing ``replace``/``drop`` behavior. The returned list is always
        in canonical order, never the order names were passed, so output is order-independent.
        """
        known = {s.name for s in self._stages}
        unknown = [n for n in requested if n not in known]
        if unknown:
            raise RegistryError(f"unknown stage(s): {unknown}. Known stages: {sorted(known)}")
        wanted = set(requested)
        return [s for s in self._stages if s.name in wanted]

    def run(
        self,
        directory: str | Path,
        out: str | Path | None = _UNSET,  # type: ignore[assignment]
        *,
        on_stage: Callable[[str], None] | None = None,
        name: str = "handbook",
        stages: Sequence[str] | None = None,
    ) -> KnowledgeSpace:
        """Run the registered stages over ``directory``; return the :class:`KnowledgeSpace`.

        When ``out`` is given (sdk.md "Execution"), the output layout is sealed to that path via
        the chosen output writer. When ``out`` is **omitted** and the pipeline was constructed with
        ``out=`` (the default output destination), the space is sealed to that constructor path
        instead (L1). When ``out`` is ``None`` (explicitly disabled) — or omitted with no
        constructor ``out=`` — the space stays in memory. ``out`` omitted is distinct from
        ``out=None`` so the CLI's ``run(dir, on_stage=...)`` path (which does not set a constructor
        ``out=`` and writes separately) is never double-written.

        ``stages=None`` (default) runs every registered stage. A subset (e.g.
        ``("walk", "parse", "chunk")``) runs **only** those stages, in their canonical pipeline
        order regardless of the order given (Feature 1, selective stage execution). A partial run
        still produces — and seals — a valid (possibly chunk-less / embedding-less) space. An
        unknown stage name raises :class:`~indx.errors.RegistryError` (exit 3) **before** any
        stage runs or any ZIP input is extracted, so nothing is processed or written.
        """
        from indx.utils.zip_input import cleanup_extracted

        # Resolve the stage subset BEFORE touching the input so an unknown stage never extracts a
        # ZIP, runs a stage, or writes a partial archive.
        selected = self._stages if stages is None else self._resolve_stage_subset(stages)

        ctx = self._new_context(directory)
        try:
            for stage in selected:
                if on_stage is not None:
                    on_stage(stage.name)
                ctx = stage.run(ctx)
                if self.strict:
                    _enforce_strict(stage.name, ctx)
            space = ctx.space
            # H1: surface parse-stage skips (files indexed with 0 chunks) on the space so the CLI
            # can warn. Mirrors the existing ``skipped_files_`` (walk) provenance field.
            space.parse_failures_ = _parse_failure_count(ctx)

            # L1: ``run()``'s own ``out`` wins; when it is omitted (the _UNSET sentinel) fall back
            # to the constructor ``out=`` as the default output destination. ``out=None`` (and an
            # omitted ``out`` with no constructor default) stays in memory.
            destination = self.out if out is _UNSET else out
            if destination is not None:
                self._writer().write(space, Path(destination), name=name)
            return space
        finally:
            cleanup_extracted(ctx.tmp_root)


class IngestResult:
    """The artifacts a single-path ingest produced, ready to merge into a space (Feature 3).

    Carries the produced :class:`~indx.core.document.Document` / :class:`~indx.core.chunk.Chunk`
    / :class:`~indx.core.relation.Relation` slices plus the manifest deltas
    (``embedding_model``/``embedding_dim``) the embed-pack stage stamped, so an archive-loaded
    CRUD caller can stamp its own manifest if it was previously unembedded.

    ``parse_failures`` is the count of files the parse stage skipped (recorded as ``kind="skip"``
    parse errors on the context). A CRUD caller surfaces it as a prominent warning so files that
    were indexed with zero chunks are not silently lost (H1).
    """

    def __init__(
        self,
        *,
        documents: list[Document],
        chunks: list[Chunk],
        relations: list[Relation],
        embedding_model: str | None,
        embedding_dim: int | None,
        parse_failures: int = 0,
    ) -> None:
        self.documents = documents
        self.chunks = chunks
        self.relations = relations
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.parse_failures = parse_failures


def ingest_path(
    pipeline: DirectoryPipeline,
    root: Path,
    rel_paths: list[str] | None = None,
    *,
    store: VectorStore | None = None,
) -> IngestResult:
    """Run ``pipeline``'s stage stack over ``root`` and return the produced artifacts (Feature 3).

    Reuses the pipeline's configured component stack (parser/embedder/store from the manifest's
    ``components``). When ``store`` is given, the embed-pack stage upserts into *that* store (the
    space's rebuilt jsonl store) instead of the pipeline's own — this is how archive-loaded CRUD,
    which has no live pipeline store, points the embed at the rebuilt store. ``rel_paths``
    restricts the Walk output to specific root-relative files (CRUD ``add``/``update`` of one
    file); ``None`` means every walked file (the build path).

    The build path calls this with ``rel_paths=None`` and ``store=None``, producing byte-identical
    output to running the stages directly, so ``build`` and CRUD share exactly one ingest code
    path.
    """
    from indx.utils.zip_input import cleanup_extracted

    if store is not None:
        # Point the embed-pack stage at the caller's store (the rebuilt jsonl store). ``use``
        # rebinds the built-in pack stage to the new store and is a no-op when embed is disabled.
        pipeline.use(store=store)

    wanted: set[str] | None = None if rel_paths is None else set(rel_paths)
    ctx = pipeline._new_context(root)
    try:
        for stage in pipeline.stages():
            ctx = stage.run(ctx)
            if pipeline.strict:
                _enforce_strict(stage.name, ctx)
            # Immediately after Walk, narrow the document set to the requested rel-paths so every
            # later stage (parse/chunk/relate/enrich/pack) operates only on the ingest subset.
            if stage.name == "walk" and wanted is not None:
                ctx.space.documents_ = [d for d in ctx.space.documents_ if d.path in wanted]
        space = ctx.space
        return IngestResult(
            documents=list(space.documents_),
            chunks=list(space.chunks),
            relations=list(space.relations),
            embedding_model=space.manifest.embedding_model,
            embedding_dim=space.manifest.embedding_dim,
            parse_failures=_parse_failure_count(ctx),
        )
    finally:
        cleanup_extracted(ctx.tmp_root)


def _parse_failure_count(ctx: SpaceContext) -> int:
    """Count parse-stage per-item skips on the context (files indexed with 0 chunks — H1)."""
    return len([e for e in ctx.errors if e.kind == "skip" and (e.stage or "") == "parse"])


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
            except Exception as exc:  # per-file failure → skip (technical-spec §3.4)
                return doc.id, StageErrorRecord(
                    stage=self.name, kind="skip", item=doc.path, message=str(exc)
                )
            # Persisting to the resume cache is a best-effort side channel: a write failure
            # (disk full, read-only/unwritable --out) must not turn a cleanly-parsed document
            # into a parse skip and discard it. Mirror StageCache.get, which already treats
            # filesystem errors as a miss rather than a crash.
            with suppress(OSError):
                self._cache.put(self.name, input_hash, self._component_id, parsed.model_dump())
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

        stamp_sources(ctx)
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
