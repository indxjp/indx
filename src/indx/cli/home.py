"""`indx home` — manage the one persistent personal knowledge base (Feature 4).

A Typer sub-group (`home_app`, mounted as ``app.add_typer(home_app, name="home")``) with three
subcommands: ``path`` (print ``$INDX_HOME``), ``stats`` (SpaceStats of the home space), and
``reset`` (confirm-guarded wipe of ``space.indx`` + ``store/``). Each wears ``@_handle_errors``
so no raw traceback escapes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from indx.cli._render import console
from indx.home import home_dir, open_home, reset_home

home_app = typer.Typer(
    no_args_is_help=True,
    help="Manage the persistent personal knowledge base ($INDX_HOME or ~/.indx).",
)


def _wire(handle_errors: Callable[[Any], Any]) -> None:
    """Register the home subcommands wrapped in the app's ``_handle_errors`` funnel.

    Called once from :mod:`indx.cli.app` so the home commands share the exact error→exit-code
    mapping (typed ``IndxError``/``typer.Abort`` passthrough) as every other CLI command, without
    importing ``app.py`` here (which would be a circular import).
    """

    @home_app.command("path")
    @handle_errors
    def home_path() -> None:
        """Print the home directory ($INDX_HOME or ~/.indx)."""
        console.print(str(home_dir()))

    @home_app.command("stats")
    @handle_errors
    def home_stats(json_out: bool = typer.Option(False, "--json")) -> None:
        """SpaceStats of the home space (initializing an empty one if absent)."""
        from indx.cli.inspect import inspect_command

        if json_out:
            console.print_json(open_home().stats.model_dump_json())
        else:
            inspect_command(None)

    @home_app.command("reset")
    @handle_errors
    def home_reset(
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    ) -> None:
        """Wipe the home space (space.indx + store/). Confirm-guarded."""
        if not yes:
            typer.confirm(
                f"This permanently deletes the home knowledge base at {home_dir()}. Continue?",
                abort=True,
            )
        reset_home()
        console.print(f"[green]reset[/green] {home_dir()}")
