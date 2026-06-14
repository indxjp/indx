# Changelog

All notable changes to `indx` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [Versioning & compatibility](https://docs.indx.jp) for what
major/minor/patch mean for the CLI, the SDK, and the `.indx` artifact format.

## [Unreleased]

## [0.0.6] — 2026-06-15

A bug-fix release resolving 11 bugs surfaced by an end-to-end bug-hunt over a clean
`pip install indx` with real downloaded documents (PDFs, HTML, CSV, unicode) and live
OpenAI/Anthropic keys. Each fix ships with a regression test; validated offline and live.
No schema change.

### Fixed

- **Silent parse failures are surfaced.** A file the parser cannot decode (e.g. a binary
  PDF under the offline plaintext parser) was indexed as a 0-chunk document with no warning
  and exit 0. `build`/`add` now count parse failures (`space.parse_failures_`), warn in
  yellow, and report `parse_failures` in `--json`; non-strict exit stays 0.
- **Build-time LLM enrichment.** An explicitly-configured build LLM is now actually used to
  generate `summary`/`topics` instead of being silently discarded. Gated to an explicit
  selection only — the zero-config/air-gapped default still records `llm = none` and uses
  the deterministic offline enricher (no network).
- **Federated query with a real embedder.** Querying a composed parent no longer fails with
  `dimension mismatch` — the embedder is resolved from the children when the parent names
  none, and `compose` stamps the parent's embedding provenance from its children.
- **Offline `ask` no longer double-prints citations.** The extractive answer no longer
  embeds a `Sources:` block; the source list is rendered once by the CLI.
- **Offline `ask` extracts relevant text.** Instead of dumping raw chunk excerpts (which the
  Markdown renderer collapsed to a blank body for tables), it does deterministic
  query-relevant sentence extraction and neutralizes pipe-table rows.
- **`indx[markitdown]` parses PDFs.** The extra now installs `markitdown[pdf]`.
- **`DirectoryPipeline(out=…)`** now persists output from `run()` (was resume-cache only).
- **`query -k`** validates `k >= 1` (was empty/negative-sliced results for `-k 0` / `-k -3`).
- **No-op `add`** reports the yellow "no documents added" message instead of a contradictory
  green `+0 docs · +0 chunks`.
- **HTML via the offline parser** is tag-stripped (stdlib only) instead of indexing raw markup.
- **CJK topics** use 2-char bigram segmentation instead of one ideograph per token.

## [0.0.5] — 2026-06-14

A bug-fix release resolving 25 bugs surfaced by an end-to-end bug-hunt over the 0.0.4
feature set (granular load, composite spaces, CRUD, home DB, filtered import). Each fix
ships with a regression test; validated offline and live against real OpenAI/Anthropic.
No new features, no schema change.

### Fixed

- **Honest build provenance.** A `build` never runs an LLM/VLM, so the sealed manifest
  **and** the CLI `--json`/text summary now report `llm`/`vlm = none` instead of the
  requested slot. `DirectoryPipeline.components` exposes the slot names as actually sealed.
- **Filtered import correctness.** Path-aware exclude globs (`**/_drafts/**` now excludes
  top-level dirs, not just leaves — was bare `fnmatch` with no `**` semantics).
- **CRUD relation integrity.** `update` / re-add no longer destroys cross-document
  relation edges — relations are recomputed over the full corpus after the change.
- **Robust ingest.** Binary / non-UTF-8 files are skipped instead of poisoning retrieval;
  CJK/non-Latin documents now get non-empty topics (Unicode-aware tokenizer).
- **`compose`** can create a parent federation that does not yet exist.
- **`[enrich] metadata`** config is now honored.
- Registry `slot:model` on a non-model slot raises a typed error instead of a raw
  `TypeError`.
- App read/browse endpoints get the same path-containment guard as export.
- `ask` bad-path exit code is now consistent with `query`; `add`/`rm` no-ops warn instead
  of printing a silent green `✓ 0 docs`.
- `inspect` renders `embedding = none` (was a malformed `embedding=/None`).
- MCP `serverInfo.version` reports indx's version, not FastMCP's.
- `/api/query` no longer inlines the full embedding vector in each hit.
- The `vectors.f32` sidecar records contributing chunk ids in row order, so a reader can
  recover the row→chunk-id mapping when some chunks are unembedded.

### Docs

- README and reference docs corrected to the real **value-first** CLI argument order
  (`indx query "text" [space]`).

## [0.0.4] — 2026-06-14

Five additive features for working with knowledge spaces beyond the initial all-or-nothing
build. Everything ships on the zero-dependency offline core stack
(`plaintext / none / hash / jsonl / .indx`) — no extras required.

### Added

- **Granular per-stage load / import.** Run and persist a prefix, suffix, or subset of the
  pipeline instead of always all six stages: `build --stages parse,chunk,relate`,
  `--through <stage>`, `--from <stage>` (and `DirectoryPipeline.run(stages=…)`). A partial run
  still seals a valid (possibly chunk-less / embedding-less) archive. Load only the members you
  ask for: `read_archive(src, members=…)`, `KnowledgeSpace.load(members=…)` / `.load_part(…)`,
  and `indx inspect <archive> --part documents|chunks|relations|manifest [--json]`. Checksum and
  zip-bomb guards still cover every member that is actually read.
- **indx of indx (composite spaces).** A `.indx` manifest can now reference other `.indx`
  archives via the additive `Manifest.children` list (`ChildRef = {name, ref, sha256?}`);
  `archive` `SCHEMA_VERSION` is bumped to `"2"` and the reader stays backward-tolerant of `"1"`.
  `indx compose <parent.indx> --add child.indx` (+ SDK `add_child`/`remove_child`) edits the
  parent only; `inspect`/`query`/`stats` federate across children (`children()`, `flatten()` —
  namespaced ids, dedup, cycle-guarded). `--no-children` limits a command to the parent.
- **CRUD over indx documents.** Append, re-ingest, or delete documents in an existing `.indx`
  incrementally — `KnowledgeSpace.add()` / `.update()` / `.remove()` and `indx add` / `indx update`
  / `indx rm` — keeping relations consistent, re-stamping chunk sources, upserting/deleting
  vectors, and resealing deterministically. Works on archive-loaded spaces with no live store.
- **Home-directory mode (one permanent DB).** A persistent personal knowledge base under
  `~/.indx/` (override `$INDX_HOME`). `query` / `ask` / `add` / `rm` / `update` / `inspect`
  default to it when no path is given. New `indx ask` command (+ `KnowledgeSpace.ask` / the
  `Answer` model): retrieve top-k and either LLM-synthesize or return a deterministic extractive
  answer with citations, fully offline. New `indx home path` / `home stats` / `home reset`.
- **Conditional / filtered directory import.** Choose which files enter the space at build time
  by glob, extension, size, depth, count, and filename via a new `[walk]` config section and the
  `WalkFilter` SDK model, with matching `build` flags: `--include` / `--exclude` / `--ext` /
  `--name-glob` (repeatable), `--min-size` / `--max-size` (with `kb`/`mb`/`gb` suffixes),
  `--max-files`, `--max-depth`. CLI overrides config per field; `--dry-run` previews the
  selection (`N files, M skipped by filter`).

### Changed

- **Breaking (CLI):** `query` / `ask` take their query text **first** and the `space` argument
  **second** (now optional, so it can default to the home space): `indx query "text" [space]`.
  The previous `indx query <space> "text"` order no longer works — swap the two arguments.
  (Pre-1.0; surfaced here because it changes an existing invocation.)
- `.indx` archives are written at `schema_version "2"`; the reader loads both `"1"` and `"2"`
  (`SUPPORTED_SCHEMA_VERSIONS`). The `children` manifest field is additive and optional.
- New public SDK symbols: `Answer`, `ChildRef`, `WalkFilter` (in `indx.__all__`).

### Fixed

- A single-file `indx add` to the home space no longer persists a dangling temporary
  directory in the sealed manifest's `source_root`; it is restored to `$INDX_HOME`.
- `KnowledgeSpace.remove()` now deletes the removed chunks' vectors from the rebuilt
  store (previously a no-op because the rows were dropped first).
- A fresh `build` whose selected stage subset omits the leading `walk` stage (e.g.
  `--from chunk`) now warns on stderr that it will index no documents, instead of
  silently sealing an empty archive.

## [0.0.3] — 2026-06-09

A bug-fix and polish release. The headline fix: published wheels now actually
ship the `indx app` web UI. The release pipeline builds the Next.js bundle into
`src/indx/app/static/` before packaging and asserts it is present, so `indx app`
serves the real interface instead of a degraded placeholder.

### Fixed

- `indx app` ships its Next.js UI bundle in the published wheel. The release
  workflow now builds the bundle before `python -m build` and verifies the
  packaged artifact contains the real `index.html` and `_next/` assets.
- 41 verified bugs surfaced by the multi-agent bug swarm across the pipeline and
  CLI.
- Web app: mobile layout collapse, a proper landing page for corrupt imports,
  and recents that can be removed.
- Web app: "Start organizing" is disabled when the dry-run plan fails, with a
  hint pointing at the fix, instead of letting the user start a build that would
  fail identically.

### Added

- IndexApp journey-product polish and batched embeddings for faster builds.

## [0.0.1] — 2026-06-07

First public release on PyPI: the light pure-Python core, the full
Walk → Parse → Chunk → Relate → Enrich → Embed+Pack pipeline, the registry of swappable
typed slots, and the complete set of optional cloud/local backends. The `.indx` archive
format is at `schema_version` `"1"`. See
[Versioning & compatibility](https://docs.indx.jp) for what
major/minor/patch mean for the CLI, the SDK, and the artifact format.

### Added

- Core pipeline and the `.indx` archive format (`schema_version` `"1"`): the light
  pure-Python Walk → Parse → Chunk → Relate → Enrich → Embed+Pack flow, the registry, and
  the swappable typed slots (parser / LLM / VLM / embedder / store / writer).
- `--offline` CLI profile that selects the pure-Python, air-gapped stack (plaintext
  parser, `hash` embedder, `jsonl` store, `llm none`) without touching the documented
  cloud defaults. This is the supported way to run with zero network calls — the slot
  defaults in `src/indx/config/schema.py` are deliberately left as-is.
- `indx demo` command that produces an instant, zero-config knowledge space so the
  "magical moment" is reachable in one command.
- Per-stage build timings, emitted as structured data via a build `--json` flag, for
  local DX measurement without telemetry.
- JSONL-directory `inspect` and `query` support so a space materialized as JSONL can be
  explored without re-sealing it into a `.indx` archive.
- `CONTRIBUTING` guide and runnable `examples/` covering real use cases.
- **AI-agent connectors** (`indx.agent`): `connect(archive)` turns a portable `.indx`
  knowledge space into live tools for an agent — one method each for LangChain
  (`.langchain()` / `.langchain_retriever()`), the OpenAI Agents SDK (`.openai()`),
  Pydantic AI (`.pydantic_ai()`), and the Claude Agent SDK (`.claude()`), plus raw tool
  specs (`.openai_schema()` / `.anthropic_schema()`) and a `.call()` dispatcher for the bare
  Chat Completions / Messages API. New extras `indx[mcp]`, `indx[pydantic-ai]`,
  `indx[openai-agents]`, `indx[claude-agent]`, and the `indx[agent]` umbrella; each adapter
  is lazily imported and gated by its extra, so core stays light.
- **`indx mcp <archive>`** — serve a knowledge space over the Model Context Protocol so any
  MCP client (Claude Desktop, Cursor, the TypeScript Mastra framework) can search it with no
  Python glue. Built on FastMCP (prefers the standalone `fastmcp` package, falls back to the
  FastMCP bundled in the `mcp` SDK). Needs `indx[mcp]` (or `indx[agent]`).
- **LiteLLM backends** (`indx[litellm]`): an `litellm` LLM adapter and embedder that reach
  any provider through one interface — on-prem (Ollama, vLLM), and AWS Bedrock, Azure
  OpenAI, GCP Vertex, or any hosted model — selected by LiteLLM's `provider/model` strings
  (e.g. `--llm litellm:bedrock/anthropic.claude-3` or `--embedder litellm:azure/<deployment>`).

### Changed

- README quickstart reworked so the first copy-pasted command succeeds offline, instead
  of failing on a missing cloud extra.

### Docs

- Overview/reference pages corrected against the real code shapes (e.g. `Document.doc_type`,
  `KnowledgeSpace.documents()` as a method with the list stored under `documents_`,
  `Manifest` carrying `schema_version` + `indx_version`).
- Added this CHANGELOG and a [Versioning & compatibility](https://docs.indx.jp)
  reference chapter documenting SemVer intent and the `.indx` schema-compat policy.
