"""Bundled sample corpus and the ``indx demo`` entry point.

Ships a tiny "team handbook" corpus under :mod:`indx.demo.corpus` so ``indx demo`` can
build → inspect → query a real knowledge space with zero user input, fully offline. The
corpus is intentionally cross-linked (onboarding → engineering guide → code review →
team directory) so the relations/lineage view has something interesting to show.
"""

from __future__ import annotations
