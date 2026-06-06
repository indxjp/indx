<div align="center">

# indx

### Make directories AI-ready — not just files.

**Point indx at a folder and get back a *knowledge space*:** structure, folder lineage,
file-to-file relationships, and semantic metadata that AI agents and RAG systems can
actually reason over.

[![PyPI](https://img.shields.io/pypi/v/indx.svg)](https://pypi.org/project/indx/)
[![Python](https://img.shields.io/pypi/pyversions/indx.svg)](https://pypi.org/project/indx/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-docs.indx.jp-brightgreen.svg)](https://docs.indx.jp)

[**Documentation**](https://docs.indx.jp) ·
[**Quickstart**](https://docs.indx.jp/start/quickstart/) ·
[**Concepts**](https://docs.indx.jp/concepts/overview/) ·
[**AI agents**](https://docs.indx.jp/guides/ai-agents/) ·
[**Changelog**](CHANGELOG.md)

</div>

---

A parser turns one PDF into clean text. **indx turns an entire folder into a knowledge
space** — and keeps the things parsers throw away: the folder a file lived in, the
documents beside it, the report it continues, the contract it references, and what kind of
document it is.

> **The thesis:** most real knowledge doesn't live in a single file — it lives in the
> *arrangement* of files. indx keeps that map and hands it to your agent.

indx **composes** file parsers (Docling, Unstructured, LlamaParse, MarkItDown, …) rather
than replacing them, then layers on what they discard. Every major component — parser, LLM,
VLM, embedder, vector store, output — is a **swappable, typed slot** with a sensible
default. Open-source · Python · CLI + SDK · Apache-2.0.

## See it in one command: `indx demo`

No data, no installs, no API keys. `indx demo` builds, inspects, and queries a bundled
sample corpus, fully offline — the whole flow in a single command:

```bash
pip install indx
indx demo
```

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

> *(A trimmed, ANSI-stripped transcript of a real `indx demo` run.)*

Now point it at your own folder:

```bash
indx ./docs --out ./ai-ready.indx --offline   # index a folder, fully offline, zero extra deps
indx inspect ./ai-ready.indx                   # structure, type histogram, relation sample
indx query   ./ai-ready.indx "how do I onboard?"
indx app                                       # visual configure → build → inspect → query in the browser
```

## Why it matters: a chunk that remembers everything

A flat `parse → split → embed → store` pipeline gives you orphaned text fragments. indx
gives you chunks that carry their whole context with them. Here's a single chunk as it
appears in the readable `index.json`:

```json
{
  "id": "chunk_0481",
  "doc_id": "doc_0007",
  "position": 12,
  "text": "Enterprise data is retained for 90 days…",
  "prev_id": "chunk_0480",
  "next_id": "chunk_0482",
  "source":   { "path": "policies/data/retention.pdf", "folder": "policies/data", "type": "policy" },
  "metadata": { "topics": ["retention", "compliance"], "summary": "90-day retention rule…", "tags": ["data-retention", "gdpr"] },
  "relations": [ { "src": "chunk_0481", "dst": "legal/gdpr.md", "type": "references", "score": 1.0 } ]
}
```

It knows **where it came from** (`source`), **what it's about** (`metadata`), **what sits
next to it** (`prev_id` / `next_id`), and **what it points to** (`relations`). An agent can
filter by location or type, expand the context window around a hit, and *follow* knowledge
instead of just matching it. Ids are deterministic, so a knowledge space is diffable and
reproducible.

## How it works

indx is a pipeline of **six ordered, individually-replaceable stages** that share one
mutable `SpaceContext`:

```text
01 Walk → 02 Parse → 03 Chunk → 04 Relate → 05 Enrich → 06 Embed+Pack
```

The pipeline is a list you control: insert a stage (say, PII redaction before enrichment),
swap one, or drop one (skip Enrich when no LLM is available) without touching its neighbors.
The whole model is symmetric across the CLI and SDK:

```python
from indx import DirectoryPipeline, KnowledgeSpace

# Build (default stack is cloud-backed; needs OPENAI_API_KEY)
space = DirectoryPipeline().run("./docs", "./ai-ready")

print(space.stats)                          # counts, timings, components used
for doc in space.documents(type="contract"):
    print(doc.path, doc.topics, doc.summary)

# Re-load the portable archive anywhere — no re-processing
space = KnowledgeSpace.load("./ai-ready/handbook.indx")
hits  = space.search("data retention", k=5)
```

The four core objects you need to know — **KnowledgeSpace**, **Document**, **Chunk**,
**Relation** — are explained in [Core concepts](https://docs.indx.jp/concepts/overview/).

## Bring your own stack — no lock-in

Every slot is a typed interface with a default and zero lock-in. Mix and match by name:

| Slot | Default (cloud) | Offline core | Other built-ins |
|---|---|---|---|
| **Parser** | `docling` | `plaintext` | unstructured · llamaparse · markitdown · textract · docintel · docai |
| **LLM** | `openai:gpt-5-mini` | `none` | ollama · anthropic · litellm · vllm · azure · bedrock · vertex |
| **VLM** | `none` | `none` | gpt4o · qwen-vl · local · bedrock · azure · vertex |
| **Embedder** | `openai:text-embedding-3-small` | `hash` | bge-m3 · e5 · cohere · bedrock · azure · vertex · litellm |
| **Store** | `qdrant` | `jsonl` (no DB) | pgvector · chroma · lancedb · s3vectors · opensearch · azure-search · bigquery · vertex-vector |
| **Output** | `.indx` archive | `.indx` / `jsonl` | langchain · llamaindex |

```bash
indx ./docs --out ./ai-ready --offline             # zero-dependency core: plaintext → hash → jsonl → .indx
indx ./docs --out ./ai-ready --store chroma        # override a single slot; everything else keeps its default
```

**Three managed cloud profiles** wire every slot to one vendor with a single install and a
single flag:

```bash
pip install "indx[aws]"   && indx ./docs --out ./out --aws     # Textract → Bedrock → Titan → S3 Vectors
pip install "indx[azure]" && indx ./docs --out ./out --azure   # Doc Intelligence → Azure OpenAI → AI Search
pip install "indx[gcp]"   && indx ./docs --out ./out --gcp     # Document AI → Gemini → gemini-embedding → BigQuery
```

> **About the offline core:** the `hash` embedder is a deterministic hashing trick, so
> offline `query` is **keyword/lexical** retrieval, *not* semantic vector search — true
> semantic search needs a real embedder extra (e.g. `bge` or `openai`). Likewise the
> offline `enrich` step derives metadata (type, topics, tags, summary) **locally, with no
> LLM call**; LLM/VLM enrichment is opt-in via the cloud/local extras. The default
> (non-`--offline`) stack is cloud-backed — install it with `pip install "indx[cloud]"` and
> set `OPENAI_API_KEY`.

## Plug a knowledge space into an AI agent

A `.indx` archive is a portable knowledge space — carry it like a **USB drive** and plug it
into any agent framework in one line:

```python
from indx.agent import connect

kb = connect("ai-ready/handbook.indx")   # load the "USB drive"
tools = kb.openai()                       # OpenAI Agents SDK …or .langchain() / .pydantic_ai() / .claude()
```

Or serve it to any [MCP](https://modelcontextprotocol.io) client — Claude Desktop, Cursor,
or the TypeScript [Mastra](https://mastra.ai) framework — with no Python glue on the client
side:

```bash
pip install "indx[agent]"            # all framework adapters + the MCP server
indx mcp ai-ready/handbook.indx      # serve indx_search / indx_overview / indx_get_document
```

Every connector exposes the same three read-only tools — **search**, **overview**,
**get-document** — built on the same retrieval path as the CLI. See the
[AI agents guide](https://docs.indx.jp/guides/ai-agents/).

## Who it's for

- **RAG / agent engineers** who want grounded context *with relationships*, not flat chunk soup.
- **Enterprise & air-gapped platform teams** that need fully local, auditable, reproducible
  ingestion across large on-prem document estates — no byte leaves the network.
- **OSS developers & integrators** who want a composable, no-lock-in library they can extend
  with their own parser, store, or output.
- **Researchers** turning archives of papers, datasets, and notes into a navigable, citable,
  shareable knowledge graph.

> indx is a **build-time knowledge layer, not a runtime framework.** It produces the
> portable archive that LangChain, LlamaIndex, agents, and vector DBs *consume* — use it
> *with* them, not instead of them. See the [comparison](https://docs.indx.jp/about/comparison/).

## Status

Alpha (`0.0.1`). The zero-dependency core path (`plaintext` → `hash` → `jsonl` → `.indx`)
runs end to end and is fully air-gapped — reach it with `indx demo` or `--offline`. The
optional cloud/local backends (docling, openai, ollama, bge-m3, qdrant, plus the managed
AWS/Azure/GCP profiles, …) are implemented and selected through the registry: install the
matching extra (e.g. `pip install "indx[cloud]"`) and provide credentials to switch a slot
onto it. The `.indx` format is at `schema_version` `"1"`; public APIs may still shift before
`1.0` — see the [CHANGELOG](CHANGELOG.md) and the [docs](https://docs.indx.jp).

## Documentation

Full documentation — quickstart, concepts, the pipeline & stages, guides, and the complete
CLI/SDK reference — lives at **[docs.indx.jp](https://docs.indx.jp)**.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
nox -s tests          # fast offline suite: unit + corpus
nox -l                # list every session (integration / docker / airgap / live / record-fixtures)
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), and
[Adding a backend](https://docs.indx.jp/contributing/adding-a-backend/) to author a new slot
implementation.

## License

[Apache-2.0](LICENSE).
