# Changelog

All notable changes to `indx` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [Versioning & compatibility](https://docs.indx.jp) for what
major/minor/patch mean for the CLI, the SDK, and the `.indx` artifact format.

## [Unreleased]

Nothing yet.

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
