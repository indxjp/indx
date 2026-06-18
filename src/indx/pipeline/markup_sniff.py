"""Detect documents whose extracted text is structured markup/data, not prose (#14 N6).

The text-grade parsers (``plaintext``/``markitdown``) chunk any decodable text verbatim. For
non-prose structured formats — pandoc-AST JSON, raw notebook JSON, LaTeX source, … — that means
the index fills with tag soup (``{"t":"Str","c":"FY2025"}``, ``\\hypertarget{…}``) which silently
poisons retrieval, with none of the "try a richer parser" warning a binary gets.

This module is a conservative, deterministic *sniff*: it only flags text that carries an
unambiguous structural signature, so ordinary prose (including Markdown, which is prose with light
markup) is never flagged. It informs a non-fatal build warning — it does not change what is
indexed.
"""

from __future__ import annotations

import json
import re

# A LaTeX source signature: a documentclass preamble or pandoc's sectioning/anchor macros. Bare
# inline math (``$x$``) in prose is intentionally NOT matched — only structural document macros.
_LATEX = re.compile(r"\\(documentclass|hypertarget|usepackage)\b|\\(sub)*section\*?\{", re.I)


def looks_like_markup(text: str) -> bool:
    """True if ``text`` is structured markup/data rather than prose (high precision).

    Recognizes three unambiguous shapes and nothing else:

    * **JSON / pandoc-AST / notebook** — the text parses as a JSON object or array (pandoc's
      ``--to json`` AST and ``.ipynb`` notebooks both serialize this way);
    * **notebook JSON** that is too large to fully parse but carries the ``nbformat`` key;
    * **LaTeX source** — a documentclass preamble or pandoc sectioning/anchor macros.

    Prose — including Markdown and reStructuredText, which read as natural language — matches none
    of these, so the sniff does not flag it.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # JSON object/array (pandoc AST, notebook). Only attempt a parse when it actually opens like
    # JSON, so a prose paragraph that merely contains a brace is never parsed/flagged.
    if stripped[0] in "{[" and stripped[-1] in "}]":
        try:
            json.loads(stripped)
            return True
        except (ValueError, RecursionError):
            # Notebooks can exceed the default parse depth / be truncated by chunking — fall back
            # to the unambiguous ``nbformat`` marker that only notebook JSON carries.
            if '"nbformat"' in stripped[:4096]:
                return True
    return bool(_LATEX.search(stripped[:4096]))
