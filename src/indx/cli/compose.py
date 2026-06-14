"""`indx compose` — reference child .indx archives from a parent (Feature 2, indx-of-indx).

Edits only the PARENT manifest's ``children`` list and re-seals the parent deterministically;
child archives are validated (a missing/corrupt child → :class:`ArchiveError`, exit 4) but never
mutated. ``inspect``/``query`` then federate transparently across the composite.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.markup import escape

from indx.archive import format as fmt
from indx.archive import read_archive
from indx.cli._render import console, resolve_archive
from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.errors import ArchiveError


def compose_command(
    parent: Path,
    *,
    add: list[Path],
    remove: list[str] | None = None,
    name: list[str] | None = None,
    no_pin: bool = False,
    relative_to: Path | None = None,
) -> None:
    """Add/remove child references on a parent ``.indx`` (re-seals the parent only).

    For each ``--add`` child: resolve + validate it via :func:`read_archive` (never mutate it),
    compute the stored ``ref`` (relative to the parent archive's directory when possible, else
    absolute; ``--relative-to`` overrides the base), pin its ``sha256`` over the child file bytes
    unless ``--no-pin``, derive the ``name`` (explicit ``--name`` or the child stem), and record
    it with :meth:`KnowledgeSpace.add_child`. For each ``--remove`` NAME drop the ref. Then re-seal
    the parent. Prints a summary.
    """
    remove = list(remove or [])
    if not add and not remove:
        raise typer.BadParameter("nothing to compose: pass at least one --add or --remove")
    if name is not None and len(name) != len(add):
        raise typer.BadParameter(f"--name count ({len(name)}) must equal --add count ({len(add)})")

    # `compose` doubles as the federation CREATION step (README: `indx compose ./all.indx --add`):
    # an ABSENT parent path means "create an empty parent space here", while an existing one is
    # edited in place. resolve_archive raises ArchiveError when the path holds no .indx; treat that
    # (and a literally-missing path) as the create case, sealing a fresh empty manifest.
    try:
        parent_archive = resolve_archive(parent)
    except ArchiveError:
        if Path(parent).exists() and not Path(parent).is_dir():
            raise
        parent_archive = Path(parent)
        space = KnowledgeSpace(manifest=Manifest())
        space._bind_source(parent_archive)
    else:
        # Load the parent fully so save() re-writes a complete archive (read_archive binds source).
        space = read_archive(parent_archive)
    base = Path(relative_to) if relative_to is not None else parent_archive.parent

    names: list[str] = []
    for i, child in enumerate(add):
        child_path = resolve_archive(child)
        # Validate the child (missing/corrupt → ArchiveError, exit 4). Never mutated.
        read_archive(child_path)
        child_name = name[i] if name is not None else child_path.stem
        if child_name in names:
            raise typer.BadParameter(f"duplicate child name {child_name!r} across --add")
        names.append(child_name)
        ref = _stored_ref(child_path, base)
        sha = None if no_pin else fmt.file_sha256(child_path)
        space.add_child(child_name, ref, sha256=sha, auto_pin=not no_pin)

    removed: list[str] = []
    for nm in remove:
        if space.remove_child(nm) is not None:
            removed.append(nm)

    # Honest provenance: a composed parent is otherwise embedder-blind (components={},
    # embedding_model=None), which would make a federated search fall back to "hash" (256-dim) and
    # mismatch the children's real vectors. Stamp the parent's embedder from the first child that
    # names one when the parent has none, so flatten()'s copied manifest carries it too. Only stamp
    # when the parent is unembedded itself (it has no chunks of its own); never override an
    # existing parent embedder.
    if space.manifest.children and space.manifest.embedding_model is None:
        for child_space in space.children():
            cm = child_space.manifest
            child_embedder = cm.components.get("embedder") or cm.embedding_model
            if child_embedder:
                space.manifest.embedding_model = cm.embedding_model
                space.manifest.embedding_dim = cm.embedding_dim
                if "embedder" not in space.manifest.components:
                    space.manifest.components = {
                        **space.manifest.components,
                        "embedder": child_embedder,
                    }
                break

    space.save(str(parent_archive))

    if names:
        console.print(f"[green]+[/green] added: {escape(', '.join(names))}")
    if removed:
        console.print(f"[yellow]-[/yellow] removed: {escape(', '.join(removed))}")
    console.print(
        f"[bold]{escape(str(parent_archive))}[/bold] now references "
        f"{len(space.manifest.children)} child space(s)."
    )


def _stored_ref(child_path: Path, base: Path) -> str:
    """Express the child as a path relative to ``base`` when possible, else absolute (portable)."""
    child_abs = child_path.resolve()
    base_abs = base.resolve()
    try:
        rel = os.path.relpath(child_abs, base_abs)
    except ValueError:
        return str(child_abs)
    # A relative path that escapes upward (``..``) is still portable as long as the parent and
    # child ship together; keep it relative.
    return rel
