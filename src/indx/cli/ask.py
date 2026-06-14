"""`indx ask` — retrieve + answer over a knowledge space (Feature 4).

Routes through :meth:`KnowledgeSpace.ask <indx.core.knowledge_space.KnowledgeSpace.ask>` so the
CLI and SDK share one retrieval+synthesis path (CLI⇄SDK parity). With no ``space`` argument the
home space (``$INDX_HOME``) is the default target, so ``indx ask "how do I onboard?"`` works
against the personal knowledge base. Offline (``llm=none``) it returns a deterministic extractive
answer with citations; a configured/overridden llm synthesizes one.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape

from indx.cli._render import console, resolve_target


def ask_command(
    space: Path | None,
    question: str,
    *,
    k: int = 5,
    llm: str | None = None,
    json_out: bool = False,
) -> None:
    """Answer ``question`` over ``space`` (or the home space when ``space`` is ``None``)."""
    ks = resolve_target(space)
    if llm is not None:
        # CLI --llm overrides the manifest for THIS answer (mirrors build --llm). Records the
        # override on a copied manifest so the on-disk space is never mutated.
        ks = ks.model_copy(
            update={
                "manifest": ks.manifest.model_copy(
                    update={"components": {**ks.manifest.components, "llm": llm}}
                )
            }
        )
    answer = ks.ask(question, k=k)

    if json_out:
        console.print_json(answer.model_dump_json())
        return

    console.print(answer.answer)
    if answer.sources:
        console.print("\n[bold]Sources[/bold]")
        for i, src in enumerate(answer.sources, start=1):
            console.print(f"  [{i}] {escape(src)}")
