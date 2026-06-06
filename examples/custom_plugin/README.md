# Custom plugin example: a third-party parser

This directory shows the **"add a backend without forking"** pattern end to end. It is a
minimal, third-party-style package that adds a new `parser` slot implementation, `uppercase`,
without editing indx itself.

## Files

- `indx_uppercase_parser.py` — the backend. A plain class that *structurally* satisfies the
  parser `typing.Protocol` (`indx.parsers.base.Parser`): it has a `name` attribute and a
  `parse(path) -> ParsedDoc` method. No base class, no inheritance. It also carries a commented
  illustration of the lazy-import + `require_extra(...)` gate a vendor-backed backend would use.
- `pyproject.toml` — the one declaration that makes the class discoverable: an entry point
  under the **`indx.parsers`** group, keyed by the slot name (`uppercase`).

## How the registry resolves it

When a user selects `parser="uppercase"`, the registry resolves the slot in this order
(`src/indx/registry/registry.py`):

1. **First-party builtins** (`BUILTINS["parser"]`). Built-ins always win name collisions, so a
   plugin can never silently shadow `plaintext`, `docling`, etc.
2. **Third-party entry points** for the slot's group(s) (`ENTRY_POINT_GROUPS["parser"]` ->
   `indx.parsers`). The first matching entry point's class is loaded and instantiated.

If neither matches, you get a `RegistryError` listing the known names. A vendor SDK that is
not installed surfaces as a friendly `MissingExtraError` — but only when the slot is actually
selected, because the gate lives inside `__init__`, not at module import time.

## Trying it (illustrative)

This package is a teaching example; you do not have to install it to read it. If you *did*
want to wire it up in a throwaway environment:

```bash
pip install -e .          # from this directory; registers the `indx.parsers` entry point
indx build ./docs --parser uppercase
# or from the SDK:
#   DirectoryPipeline(parser="uppercase", embedder="hash", store="jsonl", enrich=False)
```

The module is import-clean on a bare install, so it reads (and imports) correctly even
without being installed.
