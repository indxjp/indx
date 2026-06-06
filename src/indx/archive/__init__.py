"""The .indx on-disk format: a Zip container holding manifest + graph + data shards."""

from indx.archive.reader import read_archive
from indx.archive.writer import write_archive

__all__ = ["read_archive", "write_archive"]
