"""The six ordered pipeline stages (one file each)."""

from indx.pipeline.stages.chunk import ChunkStage
from indx.pipeline.stages.enrich import EnrichStage
from indx.pipeline.stages.pack import PackStage
from indx.pipeline.stages.parse import ParseStage
from indx.pipeline.stages.relate import RelateStage
from indx.pipeline.stages.walk import WalkStage

__all__ = [
    "WalkStage",
    "ParseStage",
    "ChunkStage",
    "RelateStage",
    "EnrichStage",
    "PackStage",
]
