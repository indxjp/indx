"""Stage protocol: every pipeline step is run(ctx) -> ctx, and is individually replaceable."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indx.core.context import SpaceContext


@runtime_checkable
class Stage(Protocol):
    name: str

    def run(self, ctx: SpaceContext) -> SpaceContext: ...
