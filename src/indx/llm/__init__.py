"""Text LLM clients. Slot default = ``openai:gpt-5-mini``.

Re-exports the ``VLM`` protocol too, so ``from indx.llm import LLM, VLM`` works as the SDK
surface documents (technical-spec §8.1).
"""

from indx.llm.base import LLM
from indx.vlm.base import VLM

__all__ = ["LLM", "VLM"]
