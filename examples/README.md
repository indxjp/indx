# indx examples

Runnable, **fully offline** examples. Everything here works on a bare `pip install -e .`
with **no extras** — the heavy stacks (docling/openai/qdrant/torch/...) are not required,
because each example selects the zero-dependency core components explicitly:

| slot     | core choice | why                                              |
| -------- | ----------- | ------------------------------------------------ |
| parser   | `plaintext` | splits text on blank lines, no network           |
| enrich   | off         | `enrich=False`; the `llm`/`vlm` slots stay unused |
| embedder | `hash`      | deterministic, model-free vectors                |
| store    | `jsonl`     | brute-force cosine search, self-contained        |

This is the same stack as the `--offline` CLI profile; the product defaults
(docling / openai / qdrant) are intentionally cloud-first and are documented elsewhere.

## Contents

- **`quickstart.py`** — SDK walkthrough. Builds a `DirectoryPipeline` over the bundled
  `sample_space/` corpus in memory, prints the document/chunk/relation/embedding counts,
  lists the documents, and runs a semantic `search()` through the SDK. Run it with:

  ```bash
  .venv/bin/python examples/quickstart.py
  ```

- **`agent_quickstart.py`** — plug a knowledge space into an AI agent. Builds the offline
  space, wraps it in `connect(...)`, and exercises the framework-agnostic connector surface
  (`overview` / `search` / `get_document` / `openai_schema` / `call`) — all with **no agent
  SDK installed**. Swap the last block for one line (`kb.openai()`, `kb.langchain()`, …) once
  you `pip install "indx[agent]"`. Run it with:

  ```bash
  .venv/bin/python examples/agent_quickstart.py
  ```

- **`sample_space/`** — a tiny two-file Markdown corpus used by `quickstart.py`, so the
  example is self-contained and needs no external data.

- **`custom_plugin/`** — a minimal third-party-style **parser plugin** (`uppercase`) showing
  the `typing.Protocol` shape and the `indx.parsers` entry-point registration that lets a
  package extend indx **without forking**. See its `README.md`.

## See also

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — dev setup, the gates and nox sessions, coding
  standards, and the full "add a backend without forking" how-to.
