"""`indx <dir> --out` — resolve config, run a DirectoryPipeline, write via the chosen writer."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from indx.cli._render import console, writer_name
from indx.config import load_config
from indx.errors import MissingExtraError
from indx.pipeline import BuildPlan, DirectoryPipeline
from indx.registry import get_writer
from indx.utils.logging import configure

# A stderr-only console for hints that must not pollute stdout (e.g. the --json payload or
# a downstream pipe). The shared `console` writes to stdout; this one mirrors its theme.
_stderr_console = Console(stderr=True)

# The zero-dependency "core" stack used by --offline. Each entry fills a slot only when the
# user did not pass that slot explicitly, so explicit --parser/--llm/etc. still win. These
# mirror the CLI override keys consumed by load_config (NOT the schema defaults, which stay
# cloud by deliberate decision).
_OFFLINE_STACK: dict[str, str] = {
    "parser": "plaintext",
    "llm": "none",
    "vlm": "none",
    "embedder": "hash",
    "store": "jsonl",
    "format": ".indx",
}

# The fully-managed single-cloud stacks selected by --aws / --azure / --gcp. Like
# _OFFLINE_STACK, each entry fills a slot only when the user did NOT pass it explicitly, and is
# applied BEFORE load_config so explicit flags win and schema defaults are never mutated
# (cloud-providers-design §7). VLM stays unset (defaults to ``none``); opt in with --vlm.
_CLOUD_STACKS: dict[str, dict[str, str]] = {
    "aws": {"parser": "textract", "llm": "bedrock", "embedder": "bedrock", "store": "s3vectors"},
    "azure": {"parser": "docintel", "llm": "azure", "embedder": "azure", "store": "azure-search"},
    "gcp": {"parser": "docai", "llm": "vertex", "embedder": "vertex", "store": "bigquery"},
}


def build_command(
    directory: Path,
    out: Path,
    *,
    config: Path | None = None,
    parser: str | None = None,
    llm: str | None = None,
    vlm: str | None = None,
    embedder: str | None = None,
    store: str | None = None,
    fmt: str | None = None,
    name: str = "handbook",
    strict: bool = False,
    no_embed: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    offline: bool = False,
    cloud: str | None = None,
    json_out: bool = False,
    jobs: int | None = None,
    quiet: bool = False,
    verbose: bool = False,
) -> None:
    configure(logging.WARNING if quiet else logging.DEBUG if verbose else logging.INFO)

    overrides: dict[str, str | None] = {
        "parser": parser,
        "llm": llm,
        "vlm": vlm,
        "embedder": embedder,
        "store": store,
        "format": fmt,
    }

    # The profile presets (--offline / --aws / --azure / --gcp) are mutually exclusive: each
    # selects a different whole-stack fill, so combining two is ambiguous. Reject >1 with a
    # Typer usage error (exit code 2), same as any bad-flag combination.
    if cloud is not None and cloud not in _CLOUD_STACKS:
        raise typer.BadParameter(f"unknown cloud preset: {cloud!r}")
    selected_presets = [
        name for name, on in (("--offline", offline), (f"--{cloud}", cloud is not None)) if on
    ]
    if len(selected_presets) > 1:
        raise typer.BadParameter(
            "the stack presets are mutually exclusive; pass only one of "
            "--offline / --aws / --azure / --gcp "
            f"(got {' and '.join(selected_presets)})"
        )

    # Did the user select *any* slot explicitly? Used both to apply a preset only to unset
    # slots and to decide whether to emit the bare-default hint on MissingExtraError.
    any_slot_passed = any(v is not None for v in overrides.values())

    # --offline / --aws / --azure / --gcp: fill the chosen stack for every slot the user did
    # NOT pass explicitly. We do this BEFORE load_config so the values flow through the normal
    # override precedence; schema.py defaults are never touched.
    preset_stack = _OFFLINE_STACK if offline else _CLOUD_STACKS.get(cloud or "")
    if preset_stack is not None:
        for slot, preset_value in preset_stack.items():
            if overrides[slot] is None:
                overrides[slot] = preset_value

    cfg = load_config(config, overrides=overrides)

    # Resolve (and thereby validate) the output writer up front, before any pipeline work.
    # cfg.output.format is a free-form string; a typo would otherwise surface only at the final
    # write step — after walk/parse/enrich/embed have all run (wasted compute and, on the
    # default/cloud stacks, real API spend). get_writer raises RegistryError (exit 3) for an
    # unknown slot, so a bad --format now fails fast with the same error and exit code.
    writer_slot = writer_name(cfg.output.format)
    writer = get_writer(writer_slot)

    start = time.perf_counter()
    try:
        pipeline = DirectoryPipeline(
            config=cfg,
            parser=cfg.parser.engine,
            llm=cfg.enrich.llm,
            vlm=cfg.enrich.vlm,
            embedder=cfg.embed.model,
            store=cfg.store.backend,
            embed=not no_embed,
            strict=strict,
            resume=resume,
            jobs=jobs,
            out=out,
        )
    except MissingExtraError:
        # A cloud preset (--aws/--azure/--gcp) was used but its backends aren't installed:
        # point at the one-install story for that cloud, then re-raise unchanged.
        if cloud is not None:
            _emit_cloud_hint(cloud)
        # The default stack targets cloud backends. If the user ran the bare command (no slot
        # flags) and did not ask for any preset, point them at the one-flag escape hatch before
        # re-raising the original (contract-locked) error unchanged.
        elif not any_slot_passed and not offline:
            _emit_offline_hint()
        raise

    # --dry-run (FR-CLI-6): Walk only, print the plan, exit 0 — no model/embed work.
    if dry_run:
        _render_plan(pipeline.plan(directory))
        return

    components = {
        "parser": cfg.parser.engine,
        "llm": cfg.enrich.llm,
        "vlm": cfg.enrich.vlm,
        "embedder": cfg.embed.model,
        "store": cfg.store.backend,
        "format": cfg.output.format,
    }

    if json_out:
        _run_json(pipeline, directory, out, name, components, start, writer)
        return

    with console.status("[bold]indexing…"):
        space = pipeline.run(directory, on_stage=lambda s: console.log(f"stage: {s}"))

    writer.write(space, Path(out), name=name)
    elapsed = time.perf_counter() - start

    # Final summary (FR-CLI-7): counts, timing, components used.
    console.print(
        f"[green]✓[/green] {len(space.documents_)} docs · {len(space.chunks)} chunks · "
        f"{len(space.relations)} relations → [bold]{escape(str(out))}[/bold] "
        f"([cyan]{elapsed:.2f}s[/cyan])"
    )
    console.print(
        f"  components: parser={cfg.parser.engine} llm={cfg.enrich.llm} "
        f"embedder={cfg.embed.model} store={cfg.store.backend} format={cfg.output.format}"
    )


def _emit_offline_hint() -> None:
    """Explain that the default stack is cloud and the offline path is one flag away.

    Emitted to stderr only, so it never corrupts a piped/JSON stdout. The original
    MissingExtraError (its message is contract-locked) is re-raised by the caller unchanged.
    """
    _stderr_console.print(
        "[yellow]hint:[/yellow] the default stack targets cloud backends "
        "(needs an extra + API key)."
    )
    _stderr_console.print(
        "      a fully offline run is one flag away: add [bold]--offline[/bold] "
        "(no installs, zero-dependency core stack),"
    )
    _stderr_console.print(
        "      or install the stack with [bold]pip install indx\\[<extra>][/bold]."
    )


def _emit_cloud_hint(cloud: str) -> None:
    """Hint the one-install story for a --aws/--azure/--gcp preset whose extra is missing.

    Emitted to stderr only (never corrupts a piped/JSON stdout). The original
    MissingExtraError (its message is contract-locked) is re-raised by the caller unchanged.
    """
    _stderr_console.print(
        f"[yellow]hint:[/yellow] the [bold]--{cloud}[/bold] preset selects a fully-managed "
        f"{cloud.upper()} stack that needs its SDK."
    )
    _stderr_console.print(
        f"      install it in one step: [bold]pip install indx\\[{cloud}][/bold] "
        "(then set that cloud's standard credentials in your environment)."
    )


def _run_json(
    pipeline: DirectoryPipeline,
    directory: Path,
    out: Path,
    name: str,
    components: dict[str, str],
    start: float,
    writer: Any,
) -> None:
    """Run the build with per-stage timings and print ONE JSON object to stdout.

    Suppresses the Rich spinner/log/summary entirely. Per-stage timing is captured by
    wrapping the pipeline's ``on_stage`` callback — the pipeline calls it at the START of
    each stage, so each stage's duration runs to the next stage's start (and the final
    stage's to the build end).
    """
    timings: list[tuple[str, float]] = []  # (stage_name, start_perf_counter)

    def on_stage(stage_name: str) -> None:
        timings.append((stage_name, time.perf_counter()))

    space = pipeline.run(directory, on_stage=on_stage)
    stages_end = time.perf_counter()
    writer.write(space, Path(out), name=name)
    end = time.perf_counter()

    # Convert start-timestamps to per-stage durations: stage i runs until stage i+1 starts,
    # and the last stage runs to the end of the pipeline (BEFORE the writer), so the writer's
    # I/O time is not misattributed to the final compute stage.
    stages: list[dict[str, str | float]] = []
    for i, (stage_name, stage_start) in enumerate(timings):
        stage_end = timings[i + 1][1] if i + 1 < len(timings) else stages_end
        stages.append({"name": stage_name, "seconds": round(stage_end - stage_start, 6)})
    # Surface the writer cost as its own entry rather than dropping it.
    stages.append({"name": "write", "seconds": round(end - stages_end, 6)})

    payload = {
        "out": str(out),
        "counts": {
            "docs": len(space.documents_),
            "chunks": len(space.chunks),
            "relations": len(space.relations),
        },
        "elapsed_s": round(end - start, 6),
        "stages": stages,
        "components": components,
    }
    console.print_json(json.dumps(payload))


def _render_plan(plan: BuildPlan) -> None:
    """Print the dry-run plan: counts, folders, files, and selected components."""
    console.print(
        f"[bold]plan[/bold] {escape(str(plan.root))}: {len(plan.documents)} files · "
        f"{len(plan.folders)} folders (dry-run, nothing written)"
    )
    for doc in plan.documents:
        console.print(f"  {escape(str(doc.path))}")
    comp = plan.components
    console.print(
        f"  components: parser={escape(comp['parser'])} llm={escape(comp['llm'])} "
        f"embedder={escape(comp['embedder'])} store={escape(comp['store'])} "
        f"enrich={'on' if plan.enrich else 'off'} embed={'on' if plan.embed else 'off'}"
    )
