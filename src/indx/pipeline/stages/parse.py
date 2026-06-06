"""02 Parse — run the configured Parser slot per Document -> ParsedDoc (FR-S02)."""

from __future__ import annotations

from indx.core.context import SpaceContext
from indx.errors import StageError
from indx.parsers.base import Parser


class ParseStage:
    name = "parse"

    def __init__(self, parser: Parser) -> None:
        self.parser = parser

    def run(self, ctx: SpaceContext) -> SpaceContext:
        for doc in ctx.space.documents_:
            path = ctx.root / doc.path
            try:
                ctx.parsed[doc.id] = self.parser.parse(path)
            except Exception as exc:  # per-file failure names the file (NFR-OBS-1)
                raise StageError(self.name, str(exc), path=doc.path) from exc
        return ctx
