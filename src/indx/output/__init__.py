"""Output writer slot. Default = .indx archive; jsonl ships in core.

Re-exports the writer protocol so ``from indx.output import OutputWriter`` works as the SDK
surface documents (technical-spec §8.1). ``OutputWriter`` is an alias of the internal
``Writer`` protocol.
"""

from indx.output.base import Writer

# ``OutputWriter`` is the documented public name for the ``Writer`` protocol.
OutputWriter = Writer

__all__ = ["OutputWriter", "Writer"]
