"""`indx query` — embed a query, search the space, render hits with lineage.

Routes through :meth:`KnowledgeSpace.search <indx.core.knowledge_space.KnowledgeSpace.search>`
so the CLI and the SDK share one retrieval code path (CLI⇄SDK parity, technical-spec §7.3,
inspect-and-query.mdx). The embedder is taken from the archive manifest (pinned at build
time), guaranteeing query-time compatibility — there is no ``--embedder`` flag.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.markup import escape
from rich.table import Table

from indx.cli._render import console, load_space


def query_command(
    space_path: Path,
    text: str,
    *,
    k: int = 5,
    type_: str | None = None,
    json_out: bool = False,
) -> None:
    space = load_space(space_path)

    # Over-fetch when filtering by type so the post-filter can still return k hits.
    hits = space.search(text, k=k * 5 if type_ else k)

    rows = []
    for hit in hits:
        doc = space.document(hit.chunk.doc_id)
        # Type filtering: prefer the chunk's own Source.type, fall back to the parent
        # Document's detected type (inspect-and-query.mdx "Query flags").
        hit_type = (hit.source.type if hit.source else None) or (doc.doc_type if doc else None)
        if type_ and (hit_type or "unknown") != type_:
            continue
        source = "/".join([*(doc.lineage if doc else []), Path(doc.path).name]) if doc else "?"
        rows.append((hit, doc, source))
        if len(rows) >= k:
            break

    if json_out:
        # Serialized SearchHit[] matching the documented contract (data-models.md §SearchHit,
        # inspect-and-query.mdx): each element carries the full `chunk` (with `source`,
        # `metadata`, neighbor ids, `relations`), its `score`, and the resolved `neighbors`
        # chunk objects.
        payload = [hit.model_dump(mode="json") for hit, _doc, _source in rows]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"query: {escape(repr(text))}")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("text")
    for hit, _doc, source in rows:
        snippet = hit.chunk.text[:80].replace("\n", " ")
        table.add_row(f"{hit.score:.3f}", escape(source), escape(snippet))
    console.print(table)
