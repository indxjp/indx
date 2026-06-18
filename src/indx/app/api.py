"""The ``/api`` router for ``indx app`` (docs/app-spec.md §3).

``fastapi`` / ``starlette`` are imported **lazily inside** :func:`build_router` (never at
module top), so importing this module is safe on a core-only install — the same recipe every
adapter uses. The router is mounted under ``/api`` by :func:`indx.app.server.create_app`, so
endpoints register their paths **without** the ``/api`` prefix.

The build endpoint streams Server-Sent Events: the synchronous :class:`DirectoryPipeline` runs
in a worker thread that forwards each frame onto the event loop via ``loop.call_soon_threadsafe``
into an :class:`asyncio.Queue` (so no threadpool token is parked), and the async generator drains
it so each pipeline stage streams as it starts. The consumer polls ``request.is_disconnected()``
and a cooperative ``threading.Event`` stop flag halts the build at the next stage boundary on
disconnect. Per-stage timing reuses the math from ``cli/build.py::_run_json``.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import tempfile
import threading
import time
import zipfile
from collections.abc import AsyncIterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from indx import __version__
from indx.app.models import (
    AgentDocumentRequest,
    AgentOverviewRequest,
    AgentSearchRequest,
    BrowseEntry,
    BrowseResponse,
    BuildRequest,
    BuildSummary,
    ComponentInfo,
    ComponentsResponse,
    ConfigGetResponse,
    ConfigPutRequest,
    ConfigValidateResponse,
    DemoResponse,
    DocumentDetail,
    DryRunDocument,
    DryRunResponse,
    FrameworkInfo,
    HealthResponse,
    ImportResponse,
    InspectDocument,
    InspectResponse,
    QueryRequest,
    QueryResponse,
    RelationEdge,
    SearchResults,
    SnippetsResponse,
    SpaceOverview,
    StageTiming,
    ToolDef,
)
from indx.cli._render import load_space, writer_name
from indx.config import Config, load_config
from indx.config.loader import find_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import APIRouter

# Order in which slots are surfaced to the UI. ``output`` is a UI alias for the ``writer``
# registry slot (docs/app-spec.md §3 "GET /api/components").
_SLOT_ORDER: tuple[str, ...] = ("parser", "llm", "vlm", "embedder", "store", "output")
# UI slot name -> the registry slot it actually reads.
_REGISTRY_SLOT = {slot: slot for slot in _SLOT_ORDER} | {"output": "writer"}

# (registry-slot, backend) -> the third-party module name(s) that backend's adapter gates on
# via ``require_extra`` (the source of truth for "is this extra installed?"). An entry is
# absent for the zero-dep core backends (which are always installed). Mirrors each adapter's
# ``require_extra(..., *module_names)`` call exactly, so the UI's install badge matches what a
# build would actually require.
_BACKEND_MODULES: dict[tuple[str, str], tuple[str, ...]] = {
    ("parser", "docling"): ("docling",),
    ("parser", "unstructured"): ("unstructured",),
    ("parser", "llamaparse"): ("llama_cloud_services",),
    ("parser", "markitdown"): ("markitdown",),
    ("llm", "openai"): ("openai",),
    ("llm", "ollama"): ("ollama",),
    ("llm", "anthropic"): ("anthropic",),
    ("llm", "vllm"): ("openai",),
    ("llm", "azure"): ("openai",),
    ("llm", "litellm"): ("litellm",),
    ("vlm", "gpt4o"): ("openai",),
    ("vlm", "qwen-vl"): ("transformers", "torch"),
    ("vlm", "local"): ("httpx",),
    ("embedder", "openai"): ("openai",),
    ("embedder", "bge-m3"): ("FlagEmbedding",),
    ("embedder", "e5"): ("sentence_transformers",),
    ("embedder", "cohere"): ("cohere",),
    ("embedder", "litellm"): ("litellm",),
    ("store", "qdrant"): ("qdrant_client",),
    ("store", "pgvector"): ("psycopg",),
    ("store", "chroma"): ("chromadb",),
    ("store", "lancedb"): ("lancedb",),
    ("writer", "langchain"): ("langchain_core",),
    ("writer", "llamaindex"): ("llama_index",),
}

# The zero-dependency offline core stack, mirroring ``cli/build.py::_OFFLINE_STACK`` but keyed
# by the UI slot names (so ``format`` is exposed as ``output`` here).
_OFFLINE_STACK: dict[str, str] = {
    "parser": "plaintext",
    "llm": "none",
    "vlm": "none",
    "embedder": "hash",
    "store": "jsonl",
    "format": ".indx",
}


# Cap the number of relation edges returned by ``/inspect`` so a pathological space cannot
# produce an unbounded payload (the frontend renders a "showing N of M" notice).
_EDGE_LIMIT = 2000

# Hard cap on an uploaded file (``POST /api/import``). 512 MiB comfortably covers a real ``.indx``
# artifact or a zipped corpus while bounding what a single request can spill onto disk.
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024

# The name of the app-owned work dir (under the system temp root) every upload is written into.
# A single shared dir keeps uploads off arbitrary paths and trivially auditable.
_IMPORT_WORKDIR_PREFIX = "indx-app-import-"

# Every app-created temp dir (build outputs, demo spaces, import work dir) is tracked here and
# removed at interpreter exit. A build/demo/import each used to ``tempfile.mkdtemp`` a dir that
# was never cleaned up, so a long-lived server — or many demo clicks — leaked a growing pile of
# ``/tmp/indx-app-*`` dirs. ``_app_mkdtemp`` registers each; ``_cleanup_app_temp_dirs`` (run once
# via ``atexit``) removes them on shutdown.
_APP_TEMP_DIRS: set[str] = set()
_CLEANUP_REGISTERED = False


def _cleanup_app_temp_dirs() -> None:
    """Best-effort removal of every app-owned temp dir created this process (atexit hook)."""
    import shutil

    for path in list(_APP_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)
        _APP_TEMP_DIRS.discard(path)


def _app_mkdtemp(prefix: str) -> Path:
    """``tempfile.mkdtemp`` that registers the new dir for cleanup at process exit."""
    global _CLEANUP_REGISTERED
    path = Path(tempfile.mkdtemp(prefix=prefix))
    _APP_TEMP_DIRS.add(str(path))
    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup_app_temp_dirs)
        _CLEANUP_REGISTERED = True
    return path


# Agent framework -> (pip extra, the module name(s) that prove the adapter is importable). An
# install badge is "installed" iff ANY listed module imports (``mcp`` is satisfied by either
# ``fastmcp`` or ``mcp``). Mirrors the lazy adapters in ``indx.agent`` (connector.py).
_AGENT_FRAMEWORKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("langchain", "langchain", ("langchain_core",)),
    ("openai-agents", "openai-agents", ("agents",)),
    ("pydantic-ai", "pydantic-ai", ("pydantic_ai",)),
    ("claude-agent", "claude-agent", ("claude_agent_sdk",)),
    ("mcp", "mcp", ("fastmcp", "mcp")),
)

# Copy-paste connector snippets surfaced by ``GET /api/agent/snippets``. Each is short, correct,
# and routes through ``indx.agent.connect`` (or the ``indx`` CLI for the MCP server).
_AGENT_SNIPPETS: dict[str, str] = {
    "python": (
        "from indx.agent import connect\n\n"
        'conn = connect("space.indx")\n'
        "print(conn.overview())\n"
        'print(conn.search("your question", k=5))'
    ),
    "cli": "indx mcp space.indx",
    "langchain": (
        "from indx.agent import connect\n\n"
        'tools = connect("space.indx").langchain()  # needs indx[langchain]\n'
        "# pass `tools` to your LangChain agent / AgentExecutor"
    ),
    "llamaindex": (
        "from indx.agent import connect\n\n"
        'conn = connect("space.indx")\n'
        "# expose conn.search / conn.get_document as LlamaIndex FunctionTools"
    ),
    "mcp": (
        "from indx.agent import connect\n\n"
        'connect("space.indx").serve()  # stdio MCP server; needs indx[mcp]'
    ),
}


def _agent_frameworks() -> list[FrameworkInfo]:
    """Build the framework install badges via ``importlib.util.find_spec`` (no imports)."""
    out: list[FrameworkInfo] = []
    for name, extra, modules in _AGENT_FRAMEWORKS:
        installed = any(importlib.util.find_spec(m) is not None for m in modules)
        out.append(FrameworkInfo(name=name, extra=extra, installed=installed))
    return out


def _static_present() -> bool:
    """Whether the bundled SPA (``indx/app/static/index.html``) is packaged."""
    import importlib.resources as resources

    try:
        ref = resources.files("indx.app") / "static" / "index.html"
        return ref.is_file()
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - defensive
        return False


def _extra_installed(modules: tuple[str, ...]) -> bool:
    """True when every module backing an extra is importable (``find_spec``)."""
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _component_list(ui_slot: str) -> list[ComponentInfo]:
    """Build the ``ComponentInfo`` list for one UI slot from ``BUILTINS`` + ``EXTRAS``."""
    from indx.registry.builtins import BUILTINS, EXTRAS

    registry_slot = _REGISTRY_SLOT[ui_slot]
    out: list[ComponentInfo] = []
    for name in BUILTINS.get(registry_slot, {}):
        extra = EXTRAS.get((registry_slot, name))
        if extra is None:
            # Zero-dep core backend: always installed.
            installed = True
        else:
            # The extra is installed iff every vendor module its adapter gates on imports.
            modules = _BACKEND_MODULES.get((registry_slot, name), ())
            installed = _extra_installed(modules)
        out.append(ComponentInfo(name=name, builtin=True, extra=extra, installed=installed))
    return out


def _defaults_map() -> dict[str, str]:
    """The documented cloud defaults, keyed by UI slot (``output`` for the format)."""
    from indx.config import defaults as d

    return {
        "parser": d.DEFAULT_PARSER,
        "llm": d.DEFAULT_LLM,
        "vlm": d.DEFAULT_VLM,
        "embedder": d.DEFAULT_EMBEDDER,
        "store": d.DEFAULT_STORE,
        "output": d.DEFAULT_FORMAT,
    }


def _offline_map() -> dict[str, str]:
    """The offline core preset, keyed by UI slot (``output`` for the format)."""
    return {
        "parser": _OFFLINE_STACK["parser"],
        "llm": _OFFLINE_STACK["llm"],
        "vlm": _OFFLINE_STACK["vlm"],
        "embedder": _OFFLINE_STACK["embedder"],
        "store": _OFFLINE_STACK["store"],
        "output": _OFFLINE_STACK["format"],
    }


def _resolve_config(req: BuildRequest) -> Config:
    """Resolve slots exactly like ``cli/build.py::build_command``.

    ``--offline`` fills the zero-dep core stack for slots the user did not pass; explicit slots
    win; then ``load_config`` applies file/env/default precedence.
    """
    overrides: dict[str, str | None] = {
        "parser": req.parser,
        "llm": req.llm,
        "vlm": req.vlm,
        "embedder": req.embedder,
        "store": req.store,
        "format": req.format,
    }
    if req.offline:
        for slot, core_value in _OFFLINE_STACK.items():
            if overrides[slot] is None:
                overrides[slot] = core_value
    config_path = Path(req.config) if req.config else _config_from_env()
    return load_config(config_path, overrides=overrides)


def _config_from_env() -> Path | None:
    """The ``indx app --config <path>`` the CLI launched with, if any.

    The CLI stashes its ``--config`` in ``INDX_APP_CONFIG`` (cli/app.py) so both the build
    endpoints and the ``/config`` editor default to the same file the user named on launch,
    rather than silently falling back to ``./indx.toml``.
    """
    raw = os.environ.get("INDX_APP_CONFIG")
    return Path(raw) if raw else None


def _components_of(cfg: Config) -> dict[str, str]:
    """The resolved slot->name map used in build summaries (matches ``build_command``)."""
    return {
        "parser": cfg.parser.engine,
        "llm": cfg.enrich.llm,
        "vlm": cfg.enrich.vlm,
        "embedder": cfg.embed.model,
        "store": cfg.store.backend,
        "format": cfg.output.format,
    }


def _inject_space_collection(cfg: Config, out: Path | None) -> None:
    """Give each app build its own Qdrant collection so corpora never cross-contaminate.

    A Qdrant collection is a global namespace: every directory ever indexed into the default
    ``indx`` collection piles into one bucket, and an Ask against it blends every corpus
    (issue #24, 1b). When the build targets a known output dir, derive a per-space collection
    name from that dir's basename (e.g. ``indx-app-6mh0erv8``) and inject it into the
    ``[store.qdrant]`` sub-table — unless the user already pinned ``collection`` explicitly,
    which always wins. Only applies to the ``qdrant`` backend; other stores ignore it.
    """
    if out is None or cfg.store.backend != "qdrant":
        return
    if "collection" in cfg.store.options():  # an explicit pin wins
        return
    name = out.resolve().name
    if not name:
        return
    sub = getattr(cfg.store, "qdrant", None)
    if isinstance(sub, dict):
        sub["collection"] = name
    else:
        cfg.store.qdrant = {"collection": name}  # type: ignore[attr-defined]  # extra="allow" sub-table


def _make_pipeline(req: BuildRequest, cfg: Config, out: Path | None) -> Any:
    """Construct a :class:`DirectoryPipeline` from a resolved config (mirrors build_command).

    ``out`` may be ``None`` for a dry-run: ``plan`` walks only and never writes, so there is no
    point allocating (and then leaking) a temp output dir for it.

    The resolved ``cfg`` is threaded in as ``config=`` so the build honors the ``--config`` file
    the app launched with (and its ``[store.<backend>]`` sub-tables) rather than re-reading
    ``./indx.toml`` from the server CWD.
    """
    from indx.pipeline import DirectoryPipeline

    _inject_space_collection(cfg, out)
    return DirectoryPipeline(
        parser=cfg.parser.engine,
        llm=cfg.enrich.llm,
        vlm=cfg.enrich.vlm,
        embedder=cfg.embed.model,
        store=cfg.store.backend,
        config=cfg,
        embed=not req.no_embed,
        resume=req.resume,
        jobs=req.jobs,
        out=out,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _run_build_summary(
    pipeline: Any,
    directory: Path,
    out: Path,
    name: str,
    components: dict[str, str],
    writer_slot: str,
    on_stage: Any,
) -> BuildSummary:
    """Run the pipeline with per-stage timings (reusing ``_run_json``'s math) and write.

    ``on_stage`` is the caller's stage callback (used to stream SSE); this wraps it to also
    record start timestamps so each stage's duration runs to the next stage's start, with the
    writer surfaced as its own ``write`` entry.
    """
    from indx.registry import get_writer

    timings: list[tuple[str, float]] = []
    start = time.perf_counter()

    def timed_stage(stage_name: str) -> None:
        timings.append((stage_name, time.perf_counter()))
        if on_stage is not None:
            on_stage(stage_name)

    # ``out=None`` keeps ``run()`` in-memory so the explicit ``get_writer(...).write`` below is the
    # *single* serialization (with ``name`` authoritative). The pipeline was constructed with
    # ``out=`` (so ``--resume``'s StageCache still lives under that dir), and omitting ``out`` here
    # would seal a second, default-named (``handbook``) archive — the double-write of issue #22.
    space = pipeline.run(directory, out=None, on_stage=timed_stage)
    stages_end = time.perf_counter()
    get_writer(writer_slot).write(space, out, name=name)
    end = time.perf_counter()

    stages: list[StageTiming] = []
    for i, (stage_name, stage_start) in enumerate(timings):
        stage_end = timings[i + 1][1] if i + 1 < len(timings) else stages_end
        stages.append(StageTiming(name=stage_name, seconds=round(stage_end - stage_start, 6)))
    stages.append(StageTiming(name="write", seconds=round(end - stages_end, 6)))

    return BuildSummary(
        out=str(out),
        counts={
            "docs": len(space.documents_),
            "chunks": len(space.chunks),
            "relations": len(space.relations),
        },
        elapsed_s=round(end - start, 6),
        stages=stages,
        components=components,
    )


class _BuildCancelled(Exception):
    """Raised inside the worker's ``on_stage`` to bail at the next stage boundary on disconnect."""


def _run_build_worker(
    req: BuildRequest,
    emit: Any,
    stop: threading.Event,
) -> None:
    """Run one build (or dry-run), pushing ``(event, data)`` SSE tuples through ``emit``.

    ``emit`` is a thread-safe callback the caller uses to forward frames onto the event loop;
    ``stop`` is a cooperative cancel flag honored at each stage boundary so a client disconnect
    halts model/network/embedding work at the next stage instead of running the build to
    completion. The temp output dir (when the caller did not pin ``out``) is removed if the build
    is cancelled or fails before emitting ``done`` so disconnects don't orphan build outputs.
    """
    out: Path | None = None
    owns_out = False
    emitted_done = False
    try:
        cfg = _resolve_config(req)
        components = _components_of(cfg)
        directory = Path(req.directory)

        if req.dry_run:
            # A dry-run walks only and never writes, so it needs no output dir (allocating one
            # here would leak a temp dir on every Ingest keystroke that re-plans).
            plan = _make_pipeline(req, cfg, None).plan(directory)
            emit(
                "plan",
                DryRunResponse(
                    root=str(plan.root),
                    documents=[
                        DryRunDocument(
                            id=d.id,
                            path=d.path,
                            type=d.doc_type,
                            folder=d.folder,
                            size_bytes=d.size_bytes,
                        )
                        for d in plan.documents
                    ],
                    folders=plan.folders,
                    components=plan.components,
                    embed=plan.embed,
                    enrich=plan.enrich,
                ).model_dump(mode="json"),
            )
            return

        if req.out:
            out = Path(req.out)
        else:
            out = _app_mkdtemp("indx-app-")
            owns_out = True
        pipeline = _make_pipeline(req, cfg, out)

        emit(
            "start",
            {
                "components": components,
                "directory": str(directory),
                "out": str(out),
            },
        )

        def on_stage(stage_name: str) -> None:
            if stop.is_set():
                raise _BuildCancelled
            emit("stage", {"name": stage_name})

        summary = _run_build_summary(
            pipeline,
            directory,
            out,
            req.name,
            components,
            writer_name(cfg.output.format),
            on_stage,
        )
        emit("done", summary.model_dump(mode="json"))
        emitted_done = True
    except _BuildCancelled:
        pass  # client gone: stop quietly, the stream is already being torn down
    except Exception as exc:  # noqa: BLE001 - surface any failure as an SSE error frame
        emit("error", {"message": f"error: {exc}"})
    finally:
        # An output dir we created but never handed to the client (cancel/error before ``done``)
        # is an orphan; remove it so a cancelled or failed build leaves nothing behind.
        if owns_out and not emitted_done and out is not None:
            import shutil

            shutil.rmtree(out, ignore_errors=True)
            _APP_TEMP_DIRS.discard(str(out))
        emit(None, None)


def build_router() -> APIRouter:
    """Build and return the ``/api`` :class:`APIRouter` (fastapi imported lazily here)."""
    from fastapi import APIRouter, HTTPException, Query, Request
    from starlette.responses import FileResponse, StreamingResponse

    from indx.errors import IndxError

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, static=_static_present())

    @router.get("/components", response_model=ComponentsResponse)
    def components() -> ComponentsResponse:
        return ComponentsResponse(
            parser=_component_list("parser"),
            llm=_component_list("llm"),
            vlm=_component_list("vlm"),
            embedder=_component_list("embedder"),
            store=_component_list("store"),
            output=_component_list("output"),
            defaults=_defaults_map(),
            offline=_offline_map(),
        )

    @router.get("/config", response_model=ConfigGetResponse)
    def config_get(path: str | None = Query(default=None)) -> ConfigGetResponse:
        # Default to the file the CLI launched with (indx app --config), else ./indx.toml.
        effective = path or (str(_config_from_env()) if _config_from_env() is not None else None)
        try:
            resolved = find_config(effective)
            cfg = load_config(effective)
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ConfigGetResponse(
            path=str(resolved) if resolved is not None else None,
            config=cfg.model_dump(),
        )

    @router.post("/config/validate", response_model=ConfigValidateResponse)
    def config_validate(body: dict[str, Any]) -> ConfigValidateResponse:
        from pydantic import ValidationError

        try:
            Config.model_validate(body)
        except ValidationError as exc:
            return ConfigValidateResponse(
                valid=False, errors=[_format_error(e) for e in exc.errors()]
            )
        return ConfigValidateResponse(valid=True)

    @router.put("/config", response_model=ConfigGetResponse)
    def config_put(body: ConfigPutRequest) -> ConfigGetResponse:
        from pydantic import ValidationError

        # Top-level ``Config`` is ``extra="allow"`` (intentional, for ``[store.<backend>]``
        # sub-tables), so a typo'd/wrong-shaped body would otherwise validate, drop its unknown
        # keys, and persist a *default* config over the user's saved stack — silent data loss
        # (issue #24). Reject unknown top-level sections here, in the app layer, without
        # loosening the core schema.
        if isinstance(body.config, dict):
            unknown = set(body.config) - set(Config.model_fields)
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown config section(s): {sorted(unknown)}",
                )
        try:
            cfg = Config.model_validate(body.config)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=[_format_error(e) for e in exc.errors()],
            ) from exc
        try:
            target = _resolve_write_path(body.path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            rendered = _to_toml(_strip_none(cfg.model_dump()))
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            target.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ConfigGetResponse(path=str(target), config=cfg.model_dump())

    @router.post("/build")
    def build(req: BuildRequest, request: Request) -> StreamingResponse:
        # The synchronous pipeline runs in a dedicated worker thread; it forwards each SSE frame
        # onto the event loop via an ``asyncio.Queue`` (fed with ``call_soon_threadsafe``) so no
        # threadpool token is parked blocking on the queue and the loop is never blocked. The
        # consumer polls ``request.is_disconnected()`` between frames and sets a cooperative stop
        # flag so a client disconnect halts the build at the next stage boundary instead of
        # running model/embedding work to completion against an output nobody will receive.
        async def stream() -> AsyncIterator[str]:
            import asyncio

            loop = asyncio.get_running_loop()
            channel: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
            stop = threading.Event()

            def emit(event: str | None, data: dict[str, Any] | None) -> None:
                # Called from the worker thread; hop back onto the loop to touch the queue.
                item = None if event is None else (event, data or {})
                loop.call_soon_threadsafe(channel.put_nowait, item)

            thread = threading.Thread(target=_run_build_worker, args=(req, emit, stop), daemon=True)
            thread.start()
            try:
                while True:
                    item = await channel.get()
                    if item is None:
                        break
                    # Yield the just-dequeued frame BEFORE probing for disconnect. Starlette's
                    # ``is_disconnected`` can latch true around end-of-body (or on a spurious
                    # ``http.disconnect``); checking it first would silently drop an
                    # already-produced terminal ``done``/``error`` frame the worker emitted.
                    # Emitting first guarantees every frame reaches the client; the probe still
                    # halts the build at the next boundary on a real disconnect.
                    event, data = item
                    yield _sse(event, data)
                    if await request.is_disconnected():
                        stop.set()
                        break
            finally:
                # On any exit (disconnect, normal end, GeneratorExit) ask the worker to bail at the
                # next stage boundary so it can't keep doing real work after the stream is gone.
                stop.set()

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ``from __future__ import annotations`` stringizes the ``request`` annotation and ``Request``
    # is local to this factory (not a module global), so FastAPI's ``get_type_hints`` cannot
    # resolve it and would mistake ``request`` for a query param. Bind the real class so FastAPI
    # injects the ASGI request and skips it during parameter solving (same as ``import_upload``).
    build.__annotations__["request"] = Request

    @router.post("/dry-run", response_model=DryRunResponse)
    def dry_run(req: BuildRequest) -> DryRunResponse:
        try:
            cfg = _resolve_config(req)
            # ``plan`` walks only and never writes, so no output dir is needed (and none is
            # allocated — the non-streaming dry-run used to leak one temp dir per request).
            pipeline = _make_pipeline(req, cfg, None)
            plan = pipeline.plan(Path(req.directory))
        except (IndxError, OSError) as exc:
            # ``plan() -> _new_context()`` raises a bare stdlib ``NotADirectoryError`` /
            # ``FileNotFoundError`` (both ``OSError``) for a path that isn't a dir/zip; without
            # this they escape as a 500. ``/api/build`` and ``/api/inspect`` already return a
            # clean 4xx for the same input — this makes dry-run consistent (issue #24).
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DryRunResponse(
            root=str(plan.root),
            documents=[
                DryRunDocument(
                    id=d.id,
                    path=d.path,
                    type=d.doc_type,
                    folder=d.folder,
                    size_bytes=d.size_bytes,
                )
                for d in plan.documents
            ],
            folders=plan.folders,
            components=plan.components,
            embed=plan.embed,
            enrich=plan.enrich,
        )

    @router.get("/inspect", response_model=InspectResponse)
    def inspect(space: str = Query(...)) -> InspectResponse:
        from collections import Counter

        from pydantic import ValidationError

        try:
            ks = load_space(Path(space))
        except (IndxError, json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rel_counts = Counter(r.type.value for r in ks.relations)
        # The graph payload: one edge per relation, capped so a pathological space can't blow up
        # the response (the histogram above stays the full count for the legend).
        edges = [
            RelationEdge(source_id=r.src, target_id=r.dst, type=r.type.value)
            for r in ks.relations[:_EDGE_LIMIT]
        ]
        return InspectResponse(
            path=space,
            manifest=ks.manifest,
            stats=ks.stats,
            types=dict(ks.stats.types),
            relations=dict(rel_counts),
            edges=edges,
            edge_total=len(ks.relations),
            documents=[
                InspectDocument(
                    id=d.id,
                    type=d.doc_type,
                    path=d.path,
                    folder=d.folder,
                    topics=d.topics,
                    tags=d.tags,
                    chunks=len(ks.chunks_for(d.id)),
                )
                for d in ks.documents_
            ],
        )

    @router.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        from pydantic import ValidationError

        try:
            ks = load_space(Path(req.space))
        except (IndxError, json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Clamp k to >=1 so a negative/zero value can't silently slice from the end of the hit
        # list (``hits[:-2]``); mirrors the agent connector's ``max(int(k), 1)`` clamp.
        k = max(req.k, 1)
        # Over-fetch when filtering by type so the post-filter still returns k hits
        # (mirrors cli/query.py::query_command).
        hits = ks.search(req.text, k=k * 5 if req.type else k)
        kept = []
        for hit in hits:
            doc = ks.document(hit.chunk.doc_id)
            hit_type = (hit.source.type if hit.source else None) or (doc.doc_type if doc else None)
            if req.type and (hit_type or "unknown") != req.type:
                continue
            kept.append(hit)
            if len(kept) >= k:
                break
        # The UI only renders text/score/source, so strip the dense embedding vector from each
        # hit's chunk (and every neighbor chunk) before returning — it is a per-hit ``list[float]``
        # of the embedder width (e.g. 256 floats) that would otherwise inline as dead payload. Copy
        # at the app boundary so the shared core ``Chunk``/``SearchHit`` models are left untouched.
        stripped = [
            hit.model_copy(
                update={
                    "chunk": hit.chunk.model_copy(update={"embedding": None}),
                    "neighbors": [n.model_copy(update={"embedding": None}) for n in hit.neighbors],
                }
            )
            for hit in kept
        ]
        return QueryResponse(hits=stripped)

    # ----------------------------------------------------------------- agent
    # Surfacing ``indx.agent`` (docs/app-backend-gaps.md §agent). Constructing the connector and
    # serving overview/search/document pulls NO vendor SDKs; only ``/frameworks`` reports install
    # state via ``find_spec``.

    @router.get("/agent/frameworks", response_model=list[FrameworkInfo])
    def agent_frameworks() -> list[FrameworkInfo]:
        return _agent_frameworks()

    @router.get("/agent/tools", response_model=list[ToolDef])
    def agent_tools() -> list[ToolDef]:
        from indx.agent.schema import TOOLS

        return list(TOOLS)

    @router.get("/agent/snippets", response_model=SnippetsResponse)
    def agent_snippets() -> SnippetsResponse:
        return SnippetsResponse(**_AGENT_SNIPPETS)

    @router.post("/agent/overview", response_model=SpaceOverview)
    def agent_overview(req: AgentOverviewRequest) -> SpaceOverview:
        from indx.agent import connect

        try:
            conn = connect(Path(req.space))
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return conn.overview(req.sample)

    @router.post("/agent/search", response_model=SearchResults)
    def agent_search(req: AgentSearchRequest) -> SearchResults:
        from indx.agent import connect

        try:
            conn = connect(Path(req.space))
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return conn.search(req.text, k=req.k, doc_type=req.doc_type)

    @router.post("/agent/document")
    def agent_document(req: AgentDocumentRequest) -> DocumentDetail | dict[str, str]:
        from indx.agent import connect

        try:
            conn = connect(Path(req.space))
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        detail = conn.get_document(req.path_or_id)
        if detail is None:
            # Sentinel, not a 500: the doc simply wasn't found (mirrors connector.call).
            return {"error": f"no document matching {req.path_or_id!r}"}
        return detail

    @router.post("/demo", response_model=DemoResponse)
    def demo() -> DemoResponse:
        import importlib.resources as resources

        corpus_ref = resources.files("indx.demo") / "corpus"
        # Keep the output dir for the lifetime of the server so the UI can inspect/query it; it is
        # registered for cleanup at process exit so repeated demo builds don't leak temp dirs.
        out = _app_mkdtemp("indx-app-demo-")
        with resources.as_file(corpus_ref) as corpus_dir:
            req = BuildRequest(directory=str(corpus_dir), out=str(out), name="demo", offline=True)
            cfg = _resolve_config(req)
            components = _components_of(cfg)
            pipeline = _make_pipeline(req, cfg, out)
            summary = _run_build_summary(
                pipeline,
                Path(str(corpus_dir)),
                out,
                "demo",
                components,
                writer_name(cfg.output.format),
                None,
            )
        return DemoResponse(out=str(out), summary=summary)

    @router.get("/browse", response_model=BrowseResponse)
    def browse(path: str | None = Query(default=None)) -> BrowseResponse:
        base = Path(path).expanduser() if path else Path.cwd()
        if not base.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {base}")
        base = base.resolve()
        # Containment guard (same allow-set as ``/api/export``): the directory picker may only walk
        # the server's working-directory tree or an app-owned ``indx-app-*`` temp dir. Without this
        # a caller could enumerate any readable directory (``?path=/etc``); reject with 400 to match
        # export's traversal path rather than leak an arbitrary listing.
        if not _is_exportable_location(base):
            raise HTTPException(
                status_code=400, detail=f"refusing to browse outside the working directory: {base}"
            )
        entries: list[BrowseEntry] = []
        try:
            children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        for child in children:
            entries.append(BrowseEntry(name=child.name, path=str(child), is_dir=child.is_dir()))
        parent = str(base.parent) if base.parent != base else None
        return BrowseResponse(path=str(base), parent=parent, entries=entries)

    # ----------------------------------------------------------------- import / export
    # Portability (docs/app-backend-gaps.md §io). Export streams a guarded ``.indx`` file;
    # import saves a multipart upload under the app-owned work dir. Both stay vendor-free.

    @router.get("/export")
    def export(space: str = Query(...)) -> FileResponse:
        try:
            archive = _resolve_export_path(space)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(
            path=archive,
            media_type="application/octet-stream",
            filename=archive.name,
        )

    @router.post("/import", response_model=ImportResponse)
    async def import_upload(request: Request) -> ImportResponse:
        # Reading the multipart form drives starlette's parser, which is backed by
        # python-multipart (folded into the ``app`` extra); the ``file`` field carries the upload.
        from starlette.datastructures import UploadFile

        # Early-reject obviously-oversized requests via the declared Content-Length before parsing
        # the form, so starlette's multipart parser does not spool the whole upload to disk first.
        # The streaming loop below remains the authoritative bound for missing/under-declared sizes.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None
            if declared is not None and declared > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes",
                )

        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="missing multipart field 'file'")

        safe_name = _sanitize_filename(upload.filename)
        workdir = _import_workdir()
        # Give each upload a unique basename so two uploads that sanitize to the same name (e.g.
        # both ``corpus.zip``) can't overwrite or interleave onto a shared target — the first
        # caller's returned path would otherwise point at the second caller's bytes. ``mkstemp``
        # atomically reserves a fresh name (and creates the file) directly under the work dir.
        fd, reserved = tempfile.mkstemp(dir=workdir, suffix=f"-{safe_name}")
        os.close(fd)
        target = Path(reserved)
        # Stream the upload to disk in bounded chunks, aborting (and cleaning up) past the cap so
        # a single request can never spill an unbounded amount onto disk.
        written = 0
        try:
            with target.open("wb") as out_fh:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes",
                        )
                    out_fh.write(chunk)
        except BaseException:
            # Any abort (over-cap 413, ClientDisconnect mid-upload, OSError such as ENOSPC/EIO)
            # leaves a partial file under the shared work dir; remove it before re-raising so
            # repeated failures don't accumulate orphaned partial uploads.
            target.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        # A ``.indx`` upload is a ready-to-inspect space; anything else is raw build input.
        # Validate the archive up front rather than trusting the extension: a truncated, empty,
        # or mis-renamed ``.indx`` would otherwise be reported as a ``space``, the UI would open it
        # optimistically (success toast), and the failure would only surface one step later as a raw
        # loader error referencing the internal temp path. Mirror the reader's contract here — a
        # ``.indx`` must be a zip carrying ``manifest.json`` — and reject with a clean 400 so the
        # caller hears about it immediately and keeps the upload affordance.
        if target.suffix.lower() == ".indx":
            if not _looks_like_indx_archive(target):
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{safe_name!r} is not a valid .indx archive "
                        "(it must be a .indx export — a zip containing manifest.json)."
                    ),
                )
            kind = "space"
        else:
            kind = "raw"
        return ImportResponse(path=str(target), kind=kind)

    # ``from __future__ import annotations`` stringizes the ``request`` annotation, and ``Request``
    # is local to this factory (not in module globals), so FastAPI's ``get_type_hints`` cannot
    # resolve it and would mistake ``request`` for a query param. Bind the real class so FastAPI
    # recognizes it as the ASGI request and skips it during parameter solving.
    import_upload.__annotations__["request"] = Request

    return router


def _looks_like_indx_archive(path: Path) -> bool:
    """Cheap structural check that ``path`` is a real ``.indx`` export.

    Mirrors the archive reader's minimum contract (:func:`indx.archive.reader.read_archive`)
    without fully loading the space: the file must be a zip that carries ``manifest.json``. This
    catches the common bad uploads (truncated/empty/non-zip files, or an unrelated file renamed
    to ``.indx``) at import time, while deeper validation (schema version, checksums) still
    happens when the space is actually inspected.
    """
    from indx.archive import format as fmt

    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return fmt.MANIFEST in zf.namelist()
    except zipfile.BadZipFile:
        return False


def _format_error(err: Mapping[str, Any]) -> str:
    """Render one pydantic error dict as a stable ``loc: msg`` string."""
    loc = ".".join(str(p) for p in err.get("loc", ()))
    msg = err.get("msg", "")
    return f"{loc}: {msg}" if loc else str(msg)


def _resolve_write_path(path: str | None) -> Path:
    """Resolve and guard the ``PUT /api/config`` write target under the server CWD.

    The default is ``./indx.toml``. A relative or absolute target is rejected with a
    ``ValueError`` if it escapes the server's working-directory tree (path-traversal guard).
    """
    cwd = Path.cwd().resolve()
    if path is None:
        return cwd / "indx.toml"
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve()
    if resolved != cwd and cwd not in resolved.parents:
        raise ValueError(f"refusing to write outside the working directory: {resolved}")
    return resolved


def _is_exportable_location(resolved: Path) -> bool:
    """Whether ``resolved`` is a path the export endpoint may stream from.

    Allowed: the server's working-directory tree (where a user keeps their own spaces) **or** an
    app-owned build dir under the system temp root. Every in-app build/demo/import writes to
    ``tempfile.mkdtemp(prefix="indx-app-...")`` (api.py build/demo/import), so without the second
    branch export is unreachable for anything the app itself produced. The temp branch still
    requires an ``indx-app-`` path component, so a traversal into an unrelated ``/tmp`` file (or
    ``/etc/passwd``) stays rejected.
    """
    cwd = Path.cwd().resolve()
    if resolved == cwd or cwd in resolved.parents:
        return True
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved == temp_root or temp_root in resolved.parents:
        rel_parts = resolved.relative_to(temp_root).parts
        return any(part.startswith("indx-app-") for part in rel_parts)
    return False


def _resolve_export_path(space: str) -> Path:
    """Resolve a ``GET /api/export`` ``space`` to the ``.indx`` file to stream, guarded.

    The resolved ``space`` must sit in an exportable location (:func:`_is_exportable_location`):
    the server's working-directory tree or an app-owned build dir under the system temp root. A
    traversal path (``../../etc``) is rejected with a :class:`ValueError`. A directory is allowed
    and the single ``.indx`` inside it is located; a bare ``.indx`` file is streamed directly.
    """
    cwd = Path.cwd().resolve()
    candidate = Path(space).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve()
    if not _is_exportable_location(resolved):
        raise ValueError(f"refusing to export outside the working directory: {resolved}")
    if not resolved.exists():
        raise ValueError(f"no such space: {space}")
    if resolved.is_file():
        # Only stream actual ``.indx`` artifacts, matching the directory branch's ``*.indx``
        # constraint — never an arbitrary readable file (e.g. ``indx.toml``) under an allowed root.
        if resolved.suffix.lower() != ".indx":
            raise ValueError(f"not an .indx artifact: {space}")
        return resolved
    # A built space is a directory holding the self-contained ``.indx`` artifact.
    archives = sorted(resolved.glob("*.indx"))
    if not archives:
        raise ValueError(f"no .indx artifact found under: {space}")
    return archives[0]


@lru_cache(maxsize=1)
def _import_workdir() -> Path:
    """The app-owned work dir every upload lands in, created once per server process.

    A single private dir keeps uploads off arbitrary filesystem locations and makes the saved set
    auditable. When ``INDX_APP_IMPORT_DIR`` is unset, a fresh ``tempfile.mkdtemp`` dir is used:
    unpredictable name (no planted-symlink redirect) and mode 0700 (no cross-user reads of private
    corpora/embeddings). When the env override is set (deployment pin / test isolation), a
    pre-existing symlink is refused and the dir is locked to 0700. ``lru_cache`` keeps the chosen
    dir stable across requests for the server's lifetime.
    """
    override = os.environ.get("INDX_APP_IMPORT_DIR")
    if override:
        base = Path(override)
        if base.is_symlink():
            raise ValueError(f"refusing symlinked import dir: {base}")
        base.mkdir(parents=True, exist_ok=True)
        os.chmod(base, 0o700)
        return base.resolve()
    # No env override: a fresh app-owned temp dir, registered for cleanup at process exit. (An
    # explicit INDX_APP_IMPORT_DIR is the caller's to manage, so it is left untouched.)
    return _app_mkdtemp(_IMPORT_WORKDIR_PREFIX).resolve()


def _sanitize_filename(name: str | None) -> str:
    """Reduce an uploaded filename to a safe basename (no dirs, no traversal, no leading dots).

    Strips any path components (defeating ``../`` and absolute paths), keeps only a conservative
    character set, and falls back to ``upload`` when nothing usable remains — so the saved file
    can never escape the work dir or shadow a dotfile.
    """
    raw = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(c for c in raw if c.isalnum() or c in (".", "-", "_")).lstrip(".")
    return cleaned or "upload"


def _strip_none(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop keys whose value is ``None``.

    ``_SlotConfig`` allows passthrough options (``ConfigDict(extra="allow")``), so a posted
    config may carry an option set to JSON ``null`` that survives validation as ``None``.
    ``tomli_w`` has no representation for ``None`` and would raise; the hand-rolled fallback
    would silently coerce it to an empty string. Dropping ``None`` values keeps both
    serializers consistent (an absent option).
    """
    return {
        k: _strip_none(v) if isinstance(v, dict) else v for k, v in data.items() if v is not None
    }


def _to_toml(data: dict[str, Any]) -> str:
    """Serialize a Config dict to TOML.

    Prefers ``tomli_w`` when importable; otherwise falls back to a small hand-rolled serializer
    that handles the Config shape (str/bool/int/float/list[str]/nested table). There is no TOML
    *writer* in the stdlib (``tomllib`` only reads).
    """
    if importlib.util.find_spec("tomli_w") is not None:
        import tomli_w  # type: ignore[import-not-found, unused-ignore]

        return str(tomli_w.dumps(data))
    return _dumps_toml(data)


def _dumps_toml(data: dict[str, Any]) -> str:
    """Minimal TOML serializer for the Config shape (top-level scalars then tables)."""
    lines: list[str] = []
    _emit_table("", data, lines)
    return "\n".join(lines) + ("\n" if lines else "")


def _emit_table(prefix: str, table: dict[str, Any], lines: list[str]) -> None:
    """Emit one TOML table: scalars/arrays first, then nested sub-tables (depth-first)."""
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in table.items() if isinstance(v, dict)}
    if prefix and (scalars or not nested):
        lines.append(f"[{prefix}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_fmt_scalar(value)}")
    if scalars:
        lines.append("")
    for key, sub in nested.items():
        child = f"{prefix}.{key}" if prefix else key
        _emit_table(child, sub, lines)


def _fmt_scalar(value: Any) -> str:
    """Format a single TOML scalar/array value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _fmt_str(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)) or item is None:
                raise ValueError("unsupported config value: array items must be scalars")
        return "[" + ", ".join(_fmt_scalar(v) for v in value) + "]"
    raise ValueError(f"unsupported config value type: {type(value).__name__}")


def _fmt_str(value: str) -> str:
    """Quote a TOML basic string, escaping every char TOML forbids unescaped.

    A literal newline/return/tab — or any other C0 control char — is illegal inside a TOML
    basic string and would make the written ``indx.toml`` fail to re-parse (breaking the
    ``PUT /api/config`` → ``tomllib`` → ``load_config`` round-trip). Config values are
    user-supplied (e.g. a pasted prompt or a store URL), so we escape backslash and quote, the
    named short escapes (``\\b \\t \\n \\f \\r``), and any remaining control char as ``\\uXXXX``.
    """
    out: list[str] = []
    short = {
        "\\": "\\\\",
        '"': '\\"',
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
    }
    for ch in value:
        if ch in short:
            out.append(short[ch])
        elif ch < "\x20" or ch == "\x7f":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'
