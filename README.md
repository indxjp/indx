# indx

> **Make directories AI-ready, not just files.** Point indx at a folder and get back a
> *knowledge space*: structure, folder lineage, file-to-file relationships, and semantic
> metadata that AI agents and RAG systems can reason over. Open-source · Python · CLI + SDK
> · Apache-2.0.

### See it: `indx demo` (build → inspect → query, fully offline)

One command builds, inspects, and queries a bundled sample corpus — no user data, no
installs, no API keys. Real captured output:

```text
$ indx demo
indx demo — building a sample 'team handbook' knowledge space…

stage: walk
stage: parse
stage: chunk
stage: relate
stage: enrich
stage: embed-pack
✓ 7 docs · 7 chunks · 19 relations → /tmp/indx-demo-XXXX/demo (0.01s)
  components: parser=plaintext llm=none embedder=hash store=jsonl format=.indx

/tmp/indx-demo-XXXX/demo  schema=1 indx=0.0.1
  documents=7 chunks=7 relations=19 embeddings=7 embedding=hash/256
       Types                    Relations
  type       count        type         count
  markdown       6        references      14
  text           1        sibling          5

sample query (keyword/lexical, offline): how do I onboard?
  score  source                      text
  0.121  engineering/code-review.md  # Code Review  Code review keeps our codebase…
  0.098  people/remote-work.md       # Remote Work Policy  Acme Robotics is remote-…
  0.095  handbook/welcome.md         # Welcome to Acme Robotics  This is the Acme …

✓ that's the whole flow — built offline with keyword/lexical retrieval, no API key.
  run it on your own folder: indx ./your-docs --out ./ai-ready.indx --offline
```

> The recording above is a trimmed, ANSI-stripped transcript of an actual `indx demo` run.

```bash
pip install indx
indx demo                                    # instant: build → inspect → query a bundled sample, fully offline, no data needed
indx ./docs --out ./ai-ready.indx --offline  # index your own folder, fully offline (zero extra deps)
indx inspect ./ai-ready.indx
indx query   ./ai-ready.indx "how do I onboard?"
indx app                                     # visual, config-driven tester: build → inspect → query in the browser (pip install indx[app])
```

> The **default** stack targets cloud backends (docling parser, OpenAI LLM +
> embeddings, qdrant store) — install it with `pip install indx[cloud]` and set the
> matching API keys. `--offline` selects the zero-dependency core stack (plaintext
> parser → `hash` embedder → `jsonl` no-DB store → `.indx` archive), so every command
> above runs as-is on a bare `pip install indx` with no extras and nothing to configure.
> For a fully managed single-vendor build, three cloud profile extras wire every slot to
> that cloud's services with one install and one flag:
> `pip install "indx[aws]"` → `indx ./docs --out ./out --aws` (Textract → Bedrock → Titan → S3 Vectors),
> `pip install "indx[azure]"` → `indx ./docs --out ./out --azure` (Document Intelligence → Azure OpenAI → AI Search),
> `pip install "indx[gcp]"` → `indx ./docs --out ./out --gcp` (Document AI → Gemini → gemini-embedding → BigQuery).
>
> Note what the offline core does and doesn't do. The `hash` embedder is a deterministic
> hashing trick, so offline `query` is **keyword/lexical** retrieval, **not** semantic
> vector search — true semantic search needs a real embedder extra (e.g. `bge` or
> `openai`) selected explicitly. Likewise, the offline `enrich` step derives metadata
> (type, topics, tags, summary) **locally and without an LLM call**; LLM/VLM enrichment is
> opt-in via the cloud/local extras.

indx **composes** file parsers (Docling, Unstructured, …) rather than replacing them, then
layers on what they discard — the *arrangement* of files. Every major component (parser,
LLM, embedder, vector store, output) is a swappable, typed slot, so you can run the
cloud default stack or the fully offline core from the same CLI.

## Plug a knowledge space into an AI agent

A `.indx` archive is a portable knowledge space — carry it like a **USB drive** and plug it
into any agent framework in one line:

```python
from indx.agent import connect

kb = connect("ai-ready/handbook.indx")   # load the "USB drive"
tools = kb.openai()                       # OpenAI Agents SDK …or .langchain() / .pydantic_ai() / .claude()
```

Or serve it to any [MCP](https://modelcontextprotocol.io) client — Claude Desktop, Cursor, the
TypeScript [Mastra](https://mastra.ai) framework — with no Python glue on the client side:

```bash
pip install "indx[agent]"            # all framework adapters + the MCP server
indx mcp ai-ready/handbook.indx      # serve indx_search / indx_overview / indx_get_document
```

Every connector exposes the same three read-only tools — **search**, **overview**,
**get-document** — built on the same retrieval path as the CLI. See the
[AI agents guide](https://docs.indx.jp/guides/ai-agents/).

## Status

Alpha (`0.0.1`). The zero-dependency core path (`plaintext` parser → `hash` embedder →
`jsonl` no-DB store → `.indx` archive) runs end to end and is fully air-gapped — reach it
with `indx demo` or by adding `--offline` to any build. The optional cloud/local backends
(docling, openai, ollama, bge-m3, qdrant, plus the managed AWS/Azure/GCP profiles, …) are
implemented and selected through the registry: install the matching extra
(e.g. `pip install "indx[cloud]"`) and provide credentials to switch a slot onto it. The
`.indx` archive format is at `schema_version` `"1"`; public APIs may still shift before
`1.0` — see the [CHANGELOG](https://github.com/indxjp/indx/blob/main/CHANGELOG.md) and the
[documentation](https://docs.indx.jp).

## Documentation

Full documentation — quickstart, guides, the pipeline & stages, and the API/CLI reference —
lives at **[docs.indx.jp](https://docs.indx.jp)**.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
nox -s tests          # fast offline suite: unit + corpus
nox -l                # list every session (integration / docker / airgap / live / record-fixtures)
```

## License

Apache-2.0.
