# Getting started with indx

indx turns a directory of documents into an AI-ready knowledge space. It walks the
folder, parses each file, chunks the text, links neighbors, and (optionally) embeds the
chunks into a vector store you can query.

## The offline core

The default stack reaches for cloud services, but indx ships a fully offline core that
needs no extras: the `plaintext` parser, the deterministic `hash` embedder, and the
self-contained `jsonl` store. This example space is indexed with exactly that stack.
