# Architecture notes

The pipeline is a list of ordered, replaceable stages running over a shared context:
Walk, Parse, Chunk, Relate, Enrich, and Embed+Pack.

Every component is a slot resolved through the registry. Built-in implementations win
name collisions over third-party plugins, and heavy vendor SDKs are lazy-imported so the
core stays import-clean on a bare install.

## Determinism

Output is byte-identical regardless of worker count, because results are re-sorted before
ids are assigned. The hash embedder is deterministic across machines, which is why it
backs the offline test corpus.
