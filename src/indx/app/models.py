"""Pydantic v2 request/response models for the ``indx app`` HTTP API (docs/app-spec.md §3).

These are pure pydantic models — **no fastapi import here**, so importing this module is safe
on a core-only install. Where the spec says to reuse a core model the response embeds the real
thing (:class:`SpaceStats`, :class:`SearchHit`, :class:`Manifest`) and serializes it as-is.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from indx.agent.schema import (
    DocumentCard,
    DocumentDetail,
    Hit,
    SearchResults,
    SpaceOverview,
    ToolDef,
)
from indx.core.knowledge_space import Manifest
from indx.core.stats import SpaceStats
from indx.store.base import SearchHit

# Re-export the agent schema models so the app's responses use the real shapes and callers can
# import them from a single place (docs/app-backend-gaps.md §agent).
__all__ = [
    "DocumentCard",
    "DocumentDetail",
    "Hit",
    "SearchResults",
    "SpaceOverview",
    "ToolDef",
]

# --------------------------------------------------------------------------- health


class HealthResponse(BaseModel):
    """``GET /api/health`` — liveness + whether the SPA bundle is packaged."""

    status: str = "ok"
    version: str
    static: bool


# --------------------------------------------------------------------------- components


class ComponentInfo(BaseModel):
    """One backend within a slot, for the config editor's dropdowns."""

    name: str
    builtin: bool
    extra: str | None
    installed: bool


class ComponentsResponse(BaseModel):
    """``GET /api/components`` — every slot's backends plus the defaults/offline presets."""

    parser: list[ComponentInfo]
    llm: list[ComponentInfo]
    vlm: list[ComponentInfo]
    embedder: list[ComponentInfo]
    store: list[ComponentInfo]
    output: list[ComponentInfo]
    defaults: dict[str, str]
    offline: dict[str, str]


# --------------------------------------------------------------------------- config


class ConfigGetResponse(BaseModel):
    """``GET /api/config`` — the resolved config plus the file it came from (if any)."""

    path: str | None
    config: dict[str, Any]


class ConfigValidateResponse(BaseModel):
    """``POST /api/config/validate`` — validation outcome; never raises to the client."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class ConfigPutRequest(BaseModel):
    """``PUT /api/config`` — write a config to an ``indx.toml`` under the server CWD."""

    path: str | None = None
    config: dict[str, Any]


# --------------------------------------------------------------------------- build


class BuildRequest(BaseModel):
    """``POST /api/build`` / ``POST /api/dry-run`` request body (docs/app-spec.md §3)."""

    directory: str
    out: str | None = None
    name: str = "handbook"
    parser: str | None = None
    llm: str | None = None
    vlm: str | None = None
    embedder: str | None = None
    store: str | None = None
    format: str | None = None
    config: str | None = None
    offline: bool = False
    no_embed: bool = False
    resume: bool = False
    dry_run: bool = False
    jobs: int | None = None


class StageTiming(BaseModel):
    """One stage's wall-clock duration, mirroring ``cli/build.py::_run_json``."""

    name: str
    seconds: float


class BuildSummary(BaseModel):
    """The ``done`` SSE event payload (and the ``/api/demo`` summary)."""

    out: str
    counts: dict[str, int]
    elapsed_s: float
    stages: list[StageTiming]
    components: dict[str, str]


# --------------------------------------------------------------------------- dry-run


class DryRunDocument(BaseModel):
    """One planned document in a dry-run (``pipeline.plan``)."""

    id: str
    path: str
    type: str | None
    folder: str
    size_bytes: int


class DryRunResponse(BaseModel):
    """``POST /api/dry-run`` — the walk-only plan, no models run."""

    root: str
    documents: list[DryRunDocument]
    folders: list[str]
    components: dict[str, str]
    embed: bool
    enrich: bool


# --------------------------------------------------------------------------- inspect


class InspectDocument(BaseModel):
    """One document row in the inspect table."""

    id: str
    type: str | None
    path: str
    folder: str
    topics: list[str]
    tags: list[str]
    chunks: int


class RelationEdge(BaseModel):
    """One directed relation edge, document-to-document, for the relationship graph."""

    source_id: str
    target_id: str
    type: str


class InspectResponse(BaseModel):
    """``GET /api/inspect`` — manifest, stats, histograms, edges, and the document table."""

    path: str
    manifest: Manifest
    stats: SpaceStats
    types: dict[str, int]
    relations: dict[str, int]
    edges: list[RelationEdge]
    edge_total: int  # full relation count; ``edges`` may be capped, so the UI can report "N of M"
    documents: list[InspectDocument]


# --------------------------------------------------------------------------- query


class QueryRequest(BaseModel):
    """``POST /api/query`` request body."""

    space: str
    text: str
    k: int = 5
    type: str | None = None


class QueryResponse(BaseModel):
    """``POST /api/query`` — ranked hits, each a serialized :class:`SearchHit`."""

    hits: list[SearchHit]


# --------------------------------------------------------------------------- demo


class DemoResponse(BaseModel):
    """``POST /api/demo`` — the temp output dir plus the offline build summary."""

    out: str
    summary: BuildSummary


# --------------------------------------------------------------------------- browse


class BrowseEntry(BaseModel):
    """One directory-picker entry."""

    name: str
    path: str
    is_dir: bool


class BrowseResponse(BaseModel):
    """``GET /api/browse`` — a minimal server-side directory listing."""

    path: str
    parent: str | None
    entries: list[BrowseEntry]


# --------------------------------------------------------------------------- import/export


class ImportResponse(BaseModel):
    """``POST /api/import`` — where the upload landed under the app-owned work dir.

    ``kind`` is ``"space"`` for an uploaded ``.indx`` artifact (feed it straight to
    ``/api/inspect``) or ``"raw"`` for any other upload (a folder/zip the Ingest → build flow
    consumes). ``path`` is the server-side absolute path to the saved file.
    """

    path: str
    kind: str


# --------------------------------------------------------------------------- agent


class FrameworkInfo(BaseModel):
    """``GET /api/agent/frameworks`` — one agent framework's install badge."""

    name: str
    extra: str
    installed: bool


class SnippetsResponse(BaseModel):
    """``GET /api/agent/snippets`` — copy-paste connector snippets per framework."""

    python: str
    cli: str
    langchain: str
    llamaindex: str
    mcp: str


class AgentOverviewRequest(BaseModel):
    """``POST /api/agent/overview`` request body."""

    space: str
    sample: int = 10


class AgentSearchRequest(BaseModel):
    """``POST /api/agent/search`` request body."""

    space: str
    text: str
    k: int = 5
    doc_type: str | None = None


class AgentDocumentRequest(BaseModel):
    """``POST /api/agent/document`` request body."""

    space: str
    path_or_id: str
