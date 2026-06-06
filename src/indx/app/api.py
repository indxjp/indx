"""The ``/api`` router for ``indx app`` (docs/app-spec.md §3).

``fastapi`` / ``starlette`` are imported **lazily inside** :func:`build_router` (never at
module top), so importing this module is safe on a core-only install — the same recipe every
adapter uses. The router is mounted under ``/api`` by :func:`indx.app.server.create_app`, so
endpoints register their paths **without** the ``/api`` prefix.

The build endpoint streams Server-Sent Events: the synchronous :class:`DirectoryPipeline` runs
in a worker thread feeding a :class:`queue.Queue`, and the async generator drains it so each
pipeline stage streams as it starts. Per-stage timing reuses the math from
``cli/build.py::_run_json``.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from indx import __version__
from indx.app.models import (
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
    DryRunDocument,
    DryRunResponse,
    HealthResponse,
    InspectDocument,
    InspectResponse,
    QueryRequest,
    QueryResponse,
    StageTiming,
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
    import os

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


def _make_pipeline(req: BuildRequest, cfg: Config, out: Path) -> Any:
    """Construct a :class:`DirectoryPipeline` from a resolved config (mirrors build_command)."""
    from indx.pipeline import DirectoryPipeline

    return DirectoryPipeline(
        parser=cfg.parser.engine,
        llm=cfg.enrich.llm,
        vlm=cfg.enrich.vlm,
        embedder=cfg.embed.model,
        store=cfg.store.backend,
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

    space = pipeline.run(directory, on_stage=timed_stage)
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


def _build_events(req: BuildRequest) -> Iterator[str]:
    """Synchronous generator of SSE frames for a build, driven by a worker thread.

    The synchronous pipeline runs in a worker thread that pushes ``(event, data)`` tuples onto
    a queue; this generator drains the queue and formats each as an SSE frame. ``dry_run``
    emits a single ``plan`` event (no models run).
    """
    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def worker() -> None:
        try:
            cfg = _resolve_config(req)
            components = _components_of(cfg)
            out = Path(req.out) if req.out else Path(tempfile.mkdtemp(prefix="indx-app-"))
            directory = Path(req.directory)
            pipeline = _make_pipeline(req, cfg, out)

            if req.dry_run:
                plan = pipeline.plan(directory)
                events.put(
                    (
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
                )
                return

            events.put(
                (
                    "start",
                    {
                        "components": components,
                        "directory": str(directory),
                        "out": str(out),
                    },
                )
            )

            def on_stage(stage_name: str) -> None:
                events.put(("stage", {"name": stage_name}))

            summary = _run_build_summary(
                pipeline,
                directory,
                out,
                req.name,
                components,
                writer_name(cfg.output.format),
                on_stage,
            )
            events.put(("done", summary.model_dump(mode="json")))
        except Exception as exc:  # noqa: BLE001 - surface any failure as an SSE error frame
            events.put(("error", {"message": f"error: {exc}"}))
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = events.get()
        if item is None:
            break
        event, data = item
        yield _sse(event, data)


def build_router() -> APIRouter:
    """Build and return the ``/api`` :class:`APIRouter` (fastapi imported lazily here)."""
    from fastapi import APIRouter, HTTPException, Query
    from starlette.concurrency import iterate_in_threadpool
    from starlette.responses import StreamingResponse

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
        target.write_text(_to_toml(cfg.model_dump()), encoding="utf-8")
        return ConfigGetResponse(path=str(target), config=cfg.model_dump())

    @router.post("/build")
    def build(req: BuildRequest) -> StreamingResponse:
        # Drain the synchronous SSE generator on a worker thread so the event loop is free.
        async def stream() -> AsyncIterator[str]:
            async for frame in iterate_in_threadpool(_build_events(req)):
                yield frame

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.post("/dry-run", response_model=DryRunResponse)
    def dry_run(req: BuildRequest) -> DryRunResponse:
        try:
            cfg = _resolve_config(req)
            out = Path(req.out) if req.out else Path(tempfile.mkdtemp(prefix="indx-app-"))
            pipeline = _make_pipeline(req, cfg, out)
            plan = pipeline.plan(Path(req.directory))
        except IndxError as exc:
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

        try:
            ks = load_space(Path(space))
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rel_counts = Counter(r.type.value for r in ks.relations)
        return InspectResponse(
            path=space,
            manifest=ks.manifest,
            stats=ks.stats,
            types=dict(ks.stats.types),
            relations=dict(rel_counts),
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
        try:
            ks = load_space(Path(req.space))
        except IndxError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Over-fetch when filtering by type so the post-filter still returns k hits
        # (mirrors cli/query.py::query_command).
        hits = ks.search(req.text, k=req.k * 5 if req.type else req.k)
        kept = []
        for hit in hits:
            doc = ks.document(hit.chunk.doc_id)
            hit_type = (hit.source.type if hit.source else None) or (doc.doc_type if doc else None)
            if req.type and (hit_type or "unknown") != req.type:
                continue
            kept.append(hit)
            if len(kept) >= req.k:
                break
        return QueryResponse(hits=kept)

    @router.post("/demo", response_model=DemoResponse)
    def demo() -> DemoResponse:
        import importlib.resources as resources

        corpus_ref = resources.files("indx.demo") / "corpus"
        # Keep the output dir for the lifetime of the server so the UI can inspect/query it.
        out = Path(tempfile.mkdtemp(prefix="indx-app-demo-"))
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
        entries: list[BrowseEntry] = []
        try:
            children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        for child in children:
            entries.append(BrowseEntry(name=child.name, path=str(child), is_dir=child.is_dir()))
        parent = str(base.parent) if base.parent != base else None
        return BrowseResponse(path=str(base), parent=parent, entries=entries)

    return router


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
        return "[" + ", ".join(_fmt_scalar(v) for v in value) + "]"
    if value is None:
        return '""'
    return _fmt_str(str(value))


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
