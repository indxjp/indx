"""`indx inspect` — summarize a .indx archive: stats, type histogram, relations (FR-CLI-2)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from indx.cli._render import console, parent_only, resolve_target

# The members an `inspect --part` request may name. `manifest` is included for the CLI surface
# (it reads only the manifest member) even though it is not a `read_archive` content member.
_PARTS = ("documents", "chunks", "relations", "manifest")


def _display_path(space_path: Path | None) -> str:
    """The header label for a space: the explicit path, or the home archive when ``None`` (F4)."""
    if space_path is not None:
        return str(space_path)
    from indx.home import home_space_path

    return str(home_space_path())


def inspect_command(
    space_path: Path | None,
    *,
    json_out: bool = False,
    documents: str | None = None,
    part: str | None = None,
    no_children: bool = False,
) -> None:
    if part is not None:
        if documents is not None:
            raise typer.BadParameter("pass either --part or --documents, not both")
        _inspect_part(space_path, part, json_out=json_out)
        return

    from indx.errors import ArchiveError

    # F4: an explicit path wins; with no path the home space ($INDX_HOME) is the target.
    loaded = resolve_target(space_path)
    # Feature 2: federate over children unless --no-children. `space` is the parent (for the child
    # tree + manifest header); `view` is the federated read view used for counts/tables.
    space = parent_only(loaded) if no_children else loaded
    federate = bool(loaded.manifest.children) and not no_children

    if json_out:
        # Emit the full SpaceStats — the same object the SDK exposes as space.stats
        # (inspect-and-query.mdx "--json: the full space.stats", index-json.md §stats).
        # `space.stats` federates over children unless parent_only stripped them; a bad child
        # here MUST raise (the --json contract reads child content), surfacing as exit 4.
        console.print_json(space.stats.model_dump_json())
        return

    # Human-readable view: federate when possible, but degrade gracefully if a child can't be
    # read so a partially-shipped composite is still inspectable (the child tree below shows the
    # offending child's state). Every path that must *read* child content (--json, query) still
    # raises ArchiveError.
    if federate:
        try:
            view = loaded.flatten()
        except ArchiveError:
            # A child can't be read — fall back to a non-federating parent-only view so the
            # counts/tables still render; the child tree below flags the offending child.
            view = parent_only(loaded)
    else:
        view = space

    if documents is not None:
        wanted = documents if documents not in ("", "all") else None
        table = Table(title="Documents" + (f" (type={escape(wanted)})" if wanted else ""))
        table.add_column("id")
        table.add_column("type")
        table.add_column("path")
        table.add_column("chunks", justify="right")
        for d in view.documents_:
            if wanted and (d.doc_type or "unknown") != wanted:
                continue
            table.add_row(
                escape(d.id),
                escape(d.doc_type or "·"),
                escape(d.path),
                str(len(view.chunks_for(d.id))),
            )
        console.print(table)
        return

    m = space.manifest
    # Counts/types/relations/documents federate over the merged view; the header/manifest line
    # reflects the parent's own manifest (schema/version/embedding).
    stats = view.stats
    console.print(
        f"[bold]{escape(_display_path(space_path))}[/bold]  "
        f"schema={escape(m.schema_version)} indx={escape(m.indx_version)}"
    )
    console.print(
        f"  documents={stats.documents} chunks={stats.chunks} "
        f"relations={stats.relations} embeddings={stats.embeddings} "
        f"embedding={escape(m.embedding_model or '')}/{stats.embed_dim} "
        f"bytes_source={stats.bytes_source}"
    )

    if m.children and not no_children:
        _render_child_tree(loaded)

    if stats.types:
        type_table = Table(title="Types")
        type_table.add_column("type")
        type_table.add_column("count", justify="right")
        for dtype, count in sorted(stats.types.items()):
            type_table.add_row(escape(dtype), str(count))
        console.print(type_table)

    rel_counts = Counter(r.type.value for r in view.relations)
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
    for d in view.documents_[:50]:
        tree.add_row(
            escape("/".join([*d.lineage, Path(d.path).name])),
            escape("/".join(d.lineage) or "·"),
            escape(", ".join(d.topics[:3])),
        )
    console.print(tree)


def _render_child_tree(space: object) -> None:
    """Render the composite child tree (Feature 2), recursively, tolerant of bad children.

    A missing / mismatched / corrupt child is shown as a row state (``missing`` / ``!checksum`` /
    ``!corrupt``) instead of raising, so a partially-shipped composite is still inspectable. Every
    path that must *read* child content (federated stats/query) still raises ``ArchiveError``.
    """
    from indx.archive import format as fmt
    from indx.archive import read_archive
    from indx.core.knowledge_space import KnowledgeSpace
    from indx.errors import ArchiveError

    assert isinstance(space, KnowledgeSpace)
    table = Table(title="Children")
    table.add_column("name")
    table.add_column("ref")
    table.add_column("state")
    table.add_column("docs", justify="right")
    table.add_column("chunks", justify="right")

    def walk(node: KnowledgeSpace, depth: int) -> None:
        base = node._source_path
        for child in node.manifest.children:
            ref_path = Path(child.ref)
            indent = "  " * depth
            label = f"{indent}{child.name}"
            if ref_path.is_absolute():
                target: Path | None = ref_path
            elif base is None:
                target = None
            else:
                target = base.parent / ref_path
            if target is None or not target.is_file():
                table.add_row(escape(label), escape(child.ref), "missing", "·", "·")
                continue
            if child.sha256 is not None and fmt.file_sha256(target) != child.sha256:
                table.add_row(escape(label), escape(child.ref), "!checksum", "·", "·")
                continue
            try:
                resolved = read_archive(target)
            except ArchiveError:
                table.add_row(escape(label), escape(child.ref), "!corrupt", "·", "·")
                continue
            table.add_row(
                escape(label),
                escape(child.ref),
                "ok",
                str(len(resolved.documents_)),
                str(len(resolved.chunks)),
            )
            walk(resolved, depth + 1)

    walk(space, 0)
    console.print(table)


def _inspect_part(space_path: Path | None, part: str, *, json_out: bool) -> None:
    """Print just one member of a space (Feature 1, `inspect --part`).

    For a real ``.indx`` archive only the requested member is decompressed/validated (the
    selective ``read_archive`` path); for a jsonl output directory the whole space loads and the
    member is sliced. With no ``space_path`` the home space (``$INDX_HOME``) is the target
    (Feature 4). An unknown ``part`` value is validated by the Typer choice set at the CLI layer;
    reaching here with one anyway is a usage error (exit 2).
    """
    from indx.cli._render import load_space

    if part not in _PARTS:
        raise typer.BadParameter(f"unknown --part {part!r}; choose one of {', '.join(_PARTS)}")

    if part == "manifest":
        space = resolve_target(space_path)
        m = space.manifest
        if json_out:
            console.print_json(m.model_dump_json())
            return
        console.print(f"[bold]{escape(_display_path(space_path))}[/bold] manifest")
        console.print(f"  schema_version={escape(m.schema_version)}")
        console.print(f"  indx_version={escape(m.indx_version)}")
        console.print(f"  source_root={escape(m.source_root)}")
        console.print(f"  components={escape(str(m.components))}")
        console.print(f"  embedding_model={escape(m.embedding_model or '·')}")
        console.print(f"  embedding_dim={m.embedding_dim}")
        return

    # documents / chunks / relations — read only that member when the source is a .indx archive
    # (Feature 1, selective load); with no explicit path the home space loads in full (Feature 4).
    if space_path is None:
        space = resolve_target(space_path)
    else:
        space = load_space(space_path, members=(part,))
    if part == "documents":
        rows = [d.model_dump(mode="json") for d in space.documents_]
        if json_out:
            console.print_json(json.dumps(rows))
            return
        table = Table(title="Documents")
        table.add_column("id")
        table.add_column("type")
        table.add_column("path")
        for d in space.documents_:
            table.add_row(escape(d.id), escape(d.doc_type or "·"), escape(d.path))
        console.print(table)
        return

    if part == "chunks":
        rows = [c.model_dump(mode="json") for c in space.chunks]
        if json_out:
            console.print_json(json.dumps(rows))
            return
        table = Table(title="Chunks")
        table.add_column("id")
        table.add_column("doc_id")
        table.add_column("pos", justify="right")
        table.add_column("len", justify="right")
        for c in space.chunks:
            table.add_row(escape(c.id), escape(c.doc_id), str(c.position), str(len(c.text)))
        console.print(table)
        return

    # relations
    rows = [r.model_dump(mode="json") for r in space.relations]
    if json_out:
        console.print_json(json.dumps(rows))
        return
    table = Table(title="Relations")
    table.add_column("type")
    table.add_column("src")
    table.add_column("dst")
    table.add_column("score", justify="right")
    for r in space.relations:
        table.add_row(escape(r.type.value), escape(r.src), escape(r.dst), f"{r.score:.2f}")
    console.print(table)
