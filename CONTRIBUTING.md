# Contributing to indx

Thanks for helping build indx — the tool that turns a directory into a portable,
AI-ready knowledge space. This guide gets you from a fresh clone to a green PR, and walks
through the one workflow most contributors come for: **adding a backend without forking**.

indx is a light pure-Python core that orchestrates a fixed
**Walk → Parse → Chunk → Relate → Enrich → Embed+Pack** pipeline. Every external capability
(parser, LLM, VLM, embedder, vector store, output writer) is a **swappable typed slot**
resolved by name through a registry. Four constraints decide every change:
**local-first / air-gapped**, **no vendor lock-in**, **light core**, **determinism**. Keep
them in mind — most review feedback traces back to one of them.

## Dev setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

That installs indx plus the dev toolchain (ruff, mypy, pytest, nox). The core stays tiny;
heavy/optional backends (docling, torch, openai, qdrant, ...) are **extras** you install
only when you need them. You never need any extra to run the offline core or the test suite.

Want to see it work right now? The [`examples/`](examples/) directory has a runnable,
fully-offline quickstart:

```bash
python examples/quickstart.py
```

## The gates

Three checks gate every change. Run them locally before you push:

```bash
ruff check src tests     # lint
mypy src/indx            # strict type-check on the public API
pytest -q                # fast offline suite: unit + corpus
```

Also run the formatter check the way CI does:

```bash
ruff format --check src tests
```

All three must be green. The suite is **offline by design** — it never reaches the network
and uses the deterministic core stack (`plaintext` / `hash` / `jsonl`), so it runs the same
on any machine.

### nox sessions

[`noxfile.py`](noxfile.py) wraps the gates and the heavier suites into reproducible
sessions. Run `nox -l` to list them; the real sessions are:

| session           | what it does                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `tests`           | fast offline suite (unit + corpus); the default `pytest` run. Matrixed across 3.11/3.12/3.13. |
| `lint`            | `ruff check` + `ruff format --check` on `src` and `tests`.           |
| `typecheck`       | `mypy src/indx` (strict).                                            |
| `integration`     | real backends via docker compose (qdrant / postgres / ollama).       |
| `docker`          | packaged-wheel distribution tests in clean containers.               |
| `airgap`          | default stack with **no egress**, in a network-isolated container.   |
| `live`            | real models end to end; nightly/manual, may require network.         |
| `record-fixtures` | regenerate recorded ParsedDoc / embedding fixtures (deliberate).     |

Day to day you only need `tests`, `lint`, and `typecheck`. The rest run in CI or on demand.

```bash
nox -s tests          # fast offline suite
nox -s lint           # ruff lint + format check
nox -s typecheck      # mypy strict
nox -l                # list every session
```

## Coding standards

A few rules carry most of the review weight:

- **Never `import` a heavy/optional vendor SDK at module top level.** Lazy-import it inside
  `__init__` and gate it with `require_extra(...)` first (see the plugin how-to below).
- **Raise typed `IndxError` subclasses, never bare `ValueError`.** Library code must not
  `print()` — only the CLI layer (`src/indx/cli/`) renders, via Rich.
- **Determinism is non-negotiable.** Same inputs + same component selection ⇒ byte-identical
  `index.json`. Sort before assigning ids; thread the `seed`; pin golden files.
- **Models are Pydantic v2** and the public API is `mypy --strict` clean.

For the full picture:

- [`docs/reference/`](docs/reference/) — the "textbook": start with
  [`00-overview.md`](docs/reference/00-overview.md) (dependency philosophy) and
  [`13-tooling.md`](docs/reference/13-tooling.md) (ruff / mypy / nox), then the per-slot
  chapters.
- The docsite contributing pages —
  [`coding-standards`](docsite/src/content/docs/contributing/coding-standards.md) and
  [`adding-a-backend`](docsite/src/content/docs/contributing/adding-a-backend.md).

## Add a backend without forking (plugin how-to)

This is the core extension story: a third-party package adds a new slot implementation
*without editing indx*. It is exactly what [`examples/custom_plugin/`](examples/custom_plugin/)
demonstrates — read it alongside this section.

### 1. Implement the slot's `typing.Protocol` — structurally

Each slot has a `typing.Protocol` you satisfy **by shape**, not by subclassing. For example,
the parser contract (`indx.parsers.base.Parser`) is just:

```python
from pathlib import Path
from indx.core.parsed import Block, ParsedDoc


class UppercaseParser:
    name = "uppercase"               # the slot name users select
    version = "1"

    def parse(self, path: Path) -> ParsedDoc:
        text = path.read_text(encoding="utf-8")
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=[Block(kind="text", text=text.upper().strip(), order=0)],
        )
```

The other slots follow the same idea: an embedder exposes `name`, `dim`, and
`embed(texts) -> list[list[float]]`; a store exposes `upsert` / `search` / `delete`; etc.
See [`docs/reference/04-plugins-registry.md`](docs/reference/04-plugins-registry.md) and the
per-slot chapters (`05-parsers.md`, `07-embeddings.md`, `08-vector-stores.md`,
`09-output-writers.md`).

### 2. Lazy-import the vendor SDK inside `__init__`, gated first

If your backend wraps a heavy/optional dependency, do **not** import it at module top level —
that would break a bare core install. Import it inside `__init__`, and call
`indx.utils.lazy.require_extra(slot, name, extra, *module_names)` **first** so a missing
dependency surfaces as one actionable `MissingExtraError` only when the slot is selected:

```python
class MyVendorParser:
    name = "myvendor"

    def __init__(self) -> None:
        from indx.utils.lazy import require_extra

        require_extra("parser", "myvendor", "myvendor", "myvendor_sdk")
        import myvendor_sdk          # imported here, never at module top level

        self._client = myvendor_sdk.Client()
```

### 3. Register via an entry point in the correct group

Advertise your class under the slot's entry-point group in your package's `pyproject.toml`.
The group names are defined in `src/indx/registry/builtins.py` (`ENTRY_POINT_GROUPS` /
`STAGE_ENTRY_POINT_GROUP`):

| slot     | entry-point group                      |
| -------- | -------------------------------------- |
| parser   | `indx.parsers`                         |
| llm      | `indx.llms`                            |
| vlm      | `indx.vlms`                            |
| embedder | `indx.embedders`                       |
| store    | `indx.stores`                          |
| writer   | `indx.outputs` (legacy alias: `indx.writers`) |
| stage    | `indx.stages`                          |

```toml
# in your plugin package's pyproject.toml
[project.entry-points."indx.parsers"]
uppercase = "indx_uppercase_parser:UppercaseParser"
```

Once your package is `pip install`ed, users select it by name — `--parser uppercase` on the
CLI, or `DirectoryPipeline(parser="uppercase", ...)` from the SDK.

### 4. How the registry resolves it

When a slot name is requested, `src/indx/registry/registry.py` resolves it in this order:

1. **First-party builtins** (`BUILTINS[slot]`) — loaded lazily by `module:attr`.
2. **Third-party entry points** for the slot's group(s) (`ENTRY_POINT_GROUPS[slot]`).

**Built-ins always win name collisions**, so a plugin can never silently shadow a built-in
name. If nothing matches, you get a `RegistryError` listing the known names; a selected
backend whose extra is missing raises `MissingExtraError` (from the gate in step 2).

## PR expectations

Before you open a PR:

- **Gates green.** `ruff check src tests`, `ruff format --check src tests`, `mypy src/indx`,
  and `pytest -q` all pass (or `nox -s lint typecheck tests`).
- **Tests added.** New behavior ships with tests. Backends get unit tests; deterministic
  output paths get golden-file coverage.
- **Determinism respected.** No new nondeterminism in serialized output — sort before
  assigning ids, thread the `seed`, and update golden fixtures deliberately
  (`nox -s record-fixtures`), never by accident.
- **Constraints honored.** No heavy import at module top level; no bare `ValueError`; no
  `print()` outside the CLI layer; the offline core stays reachable.

Thanks again — clear, well-tested, deterministic contributions keep indx portable for
everyone.
