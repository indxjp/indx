"""Typer app for indx.

Surface (technical-spec §7): the bare ``indx <dir> --out <dir>`` builds a knowledge space —
there is *no* ``build`` keyword; ``inspect`` and ``query`` are the only named subcommands.
This is implemented with a default-command group that routes an unrecognized first token
(a directory path) to the hidden ``build`` command. Errors map to the documented exit codes
(§7.4): 1 runtime, 2 usage (Click), 3 config/unknown-component, 4 archive.
"""

from __future__ import annotations

import functools
import importlib.resources as resources
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
import typer
from typer.core import TyperGroup

from indx import __version__
from indx.cli._render import console
from indx.cli.build import build_command
from indx.cli.inspect import inspect_command
from indx.cli.query import query_command
from indx.errors import (
    ArchiveError,
    ConfigError,
    IndxError,
    PipelineError,
    RegistryError,
)
from indx.utils.lazy import require_extra


class DefaultCommandGroup(TyperGroup):
    """Route an unknown first token (a ``<dir>`` to build) to the hidden ``build`` command."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        # If the first token isn't an option or a known subcommand, treat it as the <dir>
        # positional of the implicit ``build`` command.
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["build", *args]
        return super().resolve_command(ctx, args)


app = typer.Typer(
    cls=DefaultCommandGroup,
    add_completion=False,
    no_args_is_help=True,
    help="indx — make directories AI-ready, not just files.",
)

_F = TypeVar("_F", bound=Callable[..., None])

# A command body that constructs its own usage error (e.g. ``--aws --gcp``) must surface as
# exit 2, not be reclassified as a runtime error. typer >=0.26 vendors its own Click
# (``typer._click``), so ``typer.BadParameter`` is no longer a subclass of the standalone
# ``click.UsageError`` — catching only the latter lets the vendored error fall through to the
# ``except Exception`` arm (exit 1). Collect every ``UsageError`` base in play (standalone +
# whichever Click this typer carries) so the passthrough is robust across versions.
_USAGE_ERRORS: tuple[type[BaseException], ...] = tuple(
    {click.UsageError, *(b for b in typer.BadParameter.__mro__ if b.__name__ == "UsageError")}
)


def _handle_errors(func: _F) -> _F:
    """Map an error to a clean stderr message + the documented exit code (§7.4).

    Exit-code contract (errors-and-exit-codes.md "CLI exit codes"):

    * ``2`` — usage error (Click ``UsageError``: bad flags/arguments). Click normally
      raises this *before* a command body runs, but a command that constructs its own
      ``UsageError`` is mapped here too.
    * ``3`` — configuration / unknown-component error (``ConfigError`` / ``RegistryError``).
    * ``4`` — archive error (missing, corrupt, or incompatible ``.indx``).
    * ``1`` — fatal pipeline/runtime error (``PipelineError`` and any other ``IndxError``,
      including a ``--strict`` skip promoted to fatal).

    Anything that is *not* an :class:`IndxError` — a raw vendor exception such as
    ``openai.AuthenticationError`` or a backend ``ConnectionError`` — would otherwise escape
    as a stack trace + exit 1. The final ``except Exception`` clause renders it as a clean
    ``error: <message>`` to stderr and exits 1, so no traceback ever leaks from the CLI.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            func(*args, **kwargs)
        except _USAGE_ERRORS:
            raise  # let Click format + exit 2; never reclassify a usage error
        except (typer.Exit, typer.Abort):
            raise  # control-flow signals (e.g. --version, dry-run): not errors to render
        except ArchiveError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(4) from exc
        except (ConfigError, RegistryError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(3) from exc
        except PipelineError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        except IndxError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        except Exception as exc:
            # A non-IndxError escapee (a raw vendor/SDK exception: missing API key, refused
            # connection, …). Render it cleanly — no traceback — and exit with the runtime
            # code 1 so the CLI never leaks a stack to the user (DX T12).
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc

    return wrapper  # type: ignore[return-value]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"indx {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """indx command-line interface."""


@app.command(hidden=True)
@_handle_errors
def build(
    directory: Path = typer.Argument(..., exists=True, help="Directory or .zip archive to index."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory."),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, help="indx.toml (default: ./indx.toml if present)."
    ),
    parser: str | None = typer.Option(None, "--parser", help="Parser slot."),
    llm: str | None = typer.Option(None, "--llm", help="Enrichment LLM ('none' to disable)."),
    vlm: str | None = typer.Option(None, "--vlm", help="Vision model."),
    embedder: str | None = typer.Option(None, "--embedder", help="Embedder slot."),
    store: str | None = typer.Option(None, "--store", help="Vector store slot."),
    fmt: str | None = typer.Option(None, "--format", help="Output writer (.indx, jsonl, …)."),
    name: str = typer.Option("handbook", "--name", help="Archive base name."),
    strict: bool = typer.Option(False, "--strict", help="Promote per-item skips to fatal."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip vectorization (graph only)."),
    resume: bool = typer.Option(False, "--resume", help="Reuse cached stage outputs."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Walk only and print the plan; run no models."
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help=(
            "Use the zero-dependency offline core stack "
            "(plaintext/none/hash/jsonl/.indx); no extras or API key needed. "
            "Retrieval is keyword/lexical (the 'hash' embedder is a hashing trick, "
            "not semantic embeddings)."
        ),
    ),
    aws: bool = typer.Option(
        False,
        "--aws",
        help=(
            "Use the fully-managed AWS stack (Textract / Bedrock / S3 Vectors); "
            "needs indx\\[aws] + AWS credentials. Mutually exclusive with the other presets."
        ),
    ),
    azure: bool = typer.Option(
        False,
        "--azure",
        help=(
            "Use the fully-managed Azure stack (Document Intelligence / Azure OpenAI / "
            "Azure AI Search); needs indx\\[azure] + Azure credentials. "
            "Mutually exclusive with the other presets."
        ),
    ),
    gcp: bool = typer.Option(
        False,
        "--gcp",
        help=(
            "Use the fully-managed GCP stack (Document AI / Vertex Gemini / BigQuery); "
            "needs indx\\[gcp] + Google ADC. Mutually exclusive with the other presets."
        ),
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit a machine-readable build summary with per-stage timings."
    ),
    jobs: int | None = typer.Option(None, "--jobs", "-j", help="Parallel workers."),
    quiet: bool = typer.Option(False, "--quiet", help="Decrease verbosity."),
    verbose: bool = typer.Option(False, "--verbose", help="Increase verbosity."),
) -> None:
    """Process a DIRECTORY into a knowledge space (FR-CLI-1)."""
    # Resolve the mutually-exclusive cloud-preset flags to a single ``cloud`` value. Picking
    # more than one is a usage error; cross-exclusion with --offline is enforced in
    # build_command (so the SDK-facing helper owns the full rule in one place).
    chosen = [name for name, on in (("aws", aws), ("azure", azure), ("gcp", gcp)) if on]
    if len(chosen) > 1:
        raise typer.BadParameter(
            "the cloud presets are mutually exclusive; pass only one of --aws / --azure / --gcp "
            f"(got {' and '.join('--' + c for c in chosen)})"
        )
    cloud = chosen[0] if chosen else None
    build_command(
        directory,
        out,
        config=config,
        parser=parser,
        llm=llm,
        vlm=vlm,
        embedder=embedder,
        store=store,
        fmt=fmt,
        name=name,
        strict=strict,
        no_embed=no_embed,
        resume=resume,
        dry_run=dry_run,
        offline=offline,
        cloud=cloud,
        json_out=json_out,
        jobs=jobs,
        quiet=quiet,
        verbose=verbose,
    )


@app.command()
@_handle_errors
def inspect(
    space: Path = typer.Argument(..., exists=True, help="A .indx archive or output directory."),
    json_out: bool = typer.Option(False, "--json", help="Emit space stats as JSON."),
    documents: str | None = typer.Option(
        None, "--documents", help="List documents (optionally filtered by type)."
    ),
) -> None:
    """Inspect a knowledge space: tree, types, relations, stats (FR-CLI-2)."""
    inspect_command(space, json_out=json_out, documents=documents)


@app.command()
@_handle_errors
def query(
    space: Path = typer.Argument(..., exists=True, help="A .indx archive or output directory."),
    text: str = typer.Argument(..., help="Query text."),
    k: int = typer.Option(5, "-k", help="Number of hits."),
    type_: str | None = typer.Option(None, "--type", help="Restrict to a document type."),
    json_out: bool = typer.Option(False, "--json", help="Emit SearchHit[] as JSON."),
) -> None:
    """Retrieve against a knowledge space and print hits with lineage (FR-CLI-3)."""
    query_command(space, text, k=k, type_=type_, json_out=json_out)


def _require_demo_corpus(corpus_dir: Path) -> None:
    """Fail cleanly if the bundled demo corpus is missing or empty at runtime.

    Under an editable install the corpus is always present, so this is a no-op there. It
    guards the *packaged* case: if a built wheel dropped the non-.py data files (a broken
    force-include / artifacts glob — see pyproject.toml), ``resources.as_file`` still yields
    a path, but it points at nothing. Without this check the build would surface as a raw
    ``FileNotFoundError``/``StopIteration`` from the directory walk; instead we raise an
    actionable typed :class:`PipelineError` so the CLI renders ``error: …`` and exits 1
    (NFR-OBS-1) rather than leaking a traceback.
    """
    if not corpus_dir.is_dir() or not any(corpus_dir.rglob("*")):
        raise PipelineError(
            "the bundled demo corpus is missing or empty "
            f"(expected sample files under {corpus_dir}). "
            "This usually means the installed wheel did not include the packaged "
            "'indx/demo/corpus' data files. Reinstall indx, or run a build on your own "
            "folder: indx ./your-docs --out ./ai-ready.indx --offline"
        )


@app.command()
@_handle_errors
def demo() -> None:
    """Build → inspect → query a bundled sample corpus, fully offline, zero config.

    Locates the packaged "team handbook" corpus (``indx.demo.corpus``), builds it OFFLINE
    into a temp directory with the zero-dependency core stack, then reuses the real
    ``inspect``/``query`` commands so what you see is exactly what you'd run on your own
    folder. No API key, no extras, no input required (FR-CLI-5 / DX Issue 5).
    """
    corpus_ref = resources.files("indx.demo") / "corpus"
    with (
        resources.as_file(corpus_ref) as corpus_dir,
        tempfile.TemporaryDirectory(prefix="indx-demo-") as tmp,
    ):
        out = Path(tmp) / "demo"

        _require_demo_corpus(corpus_dir)

        console.print(
            "[bold]indx demo[/bold] — building a sample 'team handbook' knowledge space…\n"
        )
        build_command(corpus_dir, out, offline=True, name="demo", quiet=True)

        console.print()
        inspect_command(out)

        console.print(
            "\n[bold]sample query[/bold] (keyword/lexical, offline): "
            "[italic]how do I onboard?[/italic]"
        )
        query_command(out, "how do I onboard?", k=3)

    console.print(
        "\n[green]✓[/green] that's the whole flow — built offline with keyword/lexical "
        "retrieval, no API key.\n"
        "  run it on your own folder: "
        "[bold]indx ./your-docs --out ./ai-ready.indx --offline[/bold]"
    )


def _require_app_static(static_dir: Path) -> None:
    """Warn (do NOT fail) if the bundled web SPA is absent at runtime.

    Mirrors :func:`_require_demo_corpus`, but the missing-bundle case is non-fatal: the API is
    fully usable without the front end (docs/app-spec.md §2/§5), so we print a WARNING hint and
    let the server start. The Next.js bundle is gitignored and only present after running
    ``scripts/build_webapp.sh``; an editable checkout typically lacks it.
    """
    if not (static_dir / "index.html").is_file():
        console.print(
            "[yellow]warning:[/yellow] the web UI bundle is not built; serving the API only.\n"
            "         build it with [bold]bash scripts/build_webapp.sh[/bold] "
            "(or npm --prefix webapp run build)."
        )


# NOTE: 'app' is now a known subcommand, so DefaultCommandGroup will route ``indx app`` here
# rather than treating it as a ``<dir>`` to build. A directory literally named ``app`` is the
# one edge case — index it explicitly, e.g. ``indx ./app --out ./out`` (the ``./`` prefix
# starts with a non-letter so it is never mistaken for this subcommand).
@app.command(name="app")
@_handle_errors
def app_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open a browser to the app on launch."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, help="indx.toml the editor opens with."
    ),
) -> None:
    """Launch INDX: turn a folder into an organized, AI-ready knowledge base.

    Opens a local app where you can curate your **Library**, **Ingest** a folder, let INDX
    **Organize** it into a clean, AI-ready knowledge base, and **Ask** questions over it.
    Runs on a FastAPI server (the ``indx\\[app]`` extra) and requires ``fastapi``/``uvicorn``;
    if the front-end bundle has not been built the API still serves and a hint is printed
    (the bundle is build-time only).
    """
    # Gate on the extra FIRST — a missing ``indx[app]`` raises MissingExtraError (a plain
    # IndxError → exit 1), exactly like every other extra.
    require_extra("app", "app", "app", "fastapi", "uvicorn")

    # Hand the chosen config to the server via the environment so the web editor and the build
    # endpoints default to the file the user named, not ./indx.toml (api.py:_config_from_env).
    if config is not None:
        import os

        os.environ["INDX_APP_CONFIG"] = str(config)

    static_ref = resources.files("indx.app") / "static"
    with resources.as_file(static_ref) as static_dir:
        _require_app_static(static_dir)

    from indx.app.server import create_app, serve

    # Touch create_app so an obvious import/build error surfaces before uvicorn binds.
    _ = create_app
    console.print(
        "[bold]INDX[/bold] — your folder, organized into an AI-ready knowledge base "
        "to explore and ask."
    )
    console.print(f"[bold]indx app[/bold] → http://{host}:{port}  (Ctrl-C to stop)")
    serve(host=host, port=port, open_browser=open_browser)


# NOTE: like 'app', 'mcp' is a known subcommand, so DefaultCommandGroup routes ``indx mcp``
# here rather than treating it as a ``<dir>`` to build. A directory literally named ``mcp`` is
# the edge case — index it explicitly as ``indx ./mcp --out ./out``.
@app.command(name="mcp")
@_handle_errors
def mcp_command(
    space: Path = typer.Argument(..., exists=True, help="A .indx archive or output directory."),
    name: str | None = typer.Option(
        None, "--name", help="Server name advertised to MCP clients (default: archive stem)."
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help="MCP transport: 'stdio' (default), 'sse', or 'streamable-http'.",
    ),
) -> None:
    """Serve a knowledge space over the Model Context Protocol (the universal agent connector).

    Turns a ``.indx`` archive into a live MCP endpoint exposing ``indx_search`` /
    ``indx_overview`` / ``indx_get_document``. Point Claude Desktop, Cursor, Mastra, or any
    MCP client at it — no Python glue on the client side. Requires the 'mcp' extra
    (install indx with the mcp or agent extra).
    """
    # Gate on the extra FIRST so a missing ``indx[mcp]`` raises a clean MissingExtraError
    # (exit 1), exactly like every other extra.
    require_extra("agent connector", "mcp", "mcp", "mcp")

    from rich.console import Console
    from rich.markup import escape

    from indx.agent import connect

    connector = connect(space, name=name)
    # The stdio transport speaks JSON-RPC over stdout, so the banner MUST go to stderr or it
    # corrupts the very first protocol exchange. Use stderr unconditionally (harmless for the
    # http transports too) so `indx mcp <archive>` is a drop-in MCP server for any client.
    Console(stderr=True).print(
        f"[bold]indx mcp[/bold] → serving '[cyan]{escape(connector.name)}[/cyan]' over {transport} "
        "(Ctrl-C to stop)",
    )
    connector.serve(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    app()
