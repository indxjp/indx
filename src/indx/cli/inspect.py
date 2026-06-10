"""`indx inspect` — summarize a .indx archive: stats, type histogram, relations (FR-CLI-2)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.markup import escape
from rich.table import Table

from indx.cli._render import console, load_space


def inspect_command(
    space_path: Path, *, json_out: bool = False, documents: str | None = None
) -> None:
    space = load_space(space_path)

    if json_out:
        # Emit the full SpaceStats (documents, chunks, relations, embeddings, embed_dim,
        # types, bytes_source) — the same object the SDK exposes as space.stats
        # (inspect-and-query.mdx "--json: the full space.stats", index-json.md §stats).
        console.print_json(space.stats.model_dump_json())
        return

    if documents is not None:
        wanted = documents if documents not in ("", "all") else None
        table = Table(title="Documents" + (f" (type={escape(wanted)})" if wanted else ""))
        table.add_column("id")
        table.add_column("type")
        table.add_column("path")
        table.add_column("chunks", justify="right")
        for d in space.documents_:
            if wanted and (d.doc_type or "unknown") != wanted:
                continue
            table.add_row(
                escape(d.id),
                escape(d.doc_type or "·"),
                escape(d.path),
                str(len(space.chunks_for(d.id))),
            )
        console.print(table)
        return

    m = space.manifest
    stats = space.stats  # documented SpaceStats surface (inspect-and-query.mdx)
    console.print(
        f"[bold]{escape(str(space_path))}[/bold]  "
        f"schema={escape(m.schema_version)} indx={escape(m.indx_version)}"
    )
    console.print(
        f"  documents={stats.documents} chunks={stats.chunks} "
        f"relations={stats.relations} embeddings={stats.embeddings} "
        f"embedding={escape(m.embedding_model or '')}/{stats.embed_dim} "
        f"bytes_source={stats.bytes_source}"
    )

    if stats.types:
        type_table = Table(title="Types")
        type_table.add_column("type")
        type_table.add_column("count", justify="right")
        for dtype, count in sorted(stats.types.items()):
            type_table.add_row(escape(dtype), str(count))
        console.print(type_table)

    rel_counts = Counter(r.type.value for r in space.relations)
    if rel_counts:
        rel_table = Table(title="Relations")
        rel_table.add_column("type")
        rel_table.add_column("count", justify="right")
        for rtype, count in sorted(rel_counts.items()):
            rel_table.add_row(escape(rtype), str(count))
        console.print(rel_table)

    tree = Table(title="Documents")
    tree.add_column("path")
    tree.add_column("lineage")
    tree.add_column("topics")
    for d in space.documents_[:50]:
        tree.add_row(
            escape("/".join([*d.lineage, Path(d.path).name])),
            escape("/".join(d.lineage) or "·"),
            escape(", ".join(d.topics[:3])),
        )
    console.print(tree)
