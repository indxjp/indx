"""Tests for F10 (ext filter drops extensionless files) and F12 (CJK tokenizer missing ranges)."""

from __future__ import annotations

from pathlib import PurePosixPath

from indx.pipeline.filters import WalkFilter
from indx.pipeline.stages.enrich import _tokenize

# ---------------------------------------------------------------------------
# F10: ext filter must not drop extensionless files when name predicate matches
# ---------------------------------------------------------------------------


class TestExtFilterExtensionlessFiles:
    def _filter(self) -> WalkFilter:
        return WalkFilter(ext=[".md"], name=["Dockerfile"])

    def test_extensionless_name_match_kept(self) -> None:
        """Dockerfile has no extension but matches name predicate — must be kept."""
        f = self._filter()
        assert f.keep(PurePosixPath("Dockerfile"), size_bytes=100, depth=0) is True

    def test_extensionless_no_name_match_dropped(self) -> None:
        """'random' has no extension and does not match name predicate — must be dropped."""
        f = self._filter()
        assert f.keep(PurePosixPath("random"), size_bytes=100, depth=0) is False

    def test_matching_ext_kept(self) -> None:
        """README.md has .md extension — must be kept regardless of name predicate."""
        f = self._filter()
        assert f.keep(PurePosixPath("README.md"), size_bytes=100, depth=0) is True


# ---------------------------------------------------------------------------
# F12: CJK tokenizer must handle half-width Katakana and Hangul Jamo
# ---------------------------------------------------------------------------


class TestCjkTokenizerMissingRanges:
    def test_half_width_katakana(self) -> None:
        """Half-width Katakana U+FF66-FF68 (ｦｧｨ) must produce tokens."""
        tokens = _tokenize("ｦｧｨ")
        assert len(tokens) > 0, "half-width Katakana returned no tokens"

    def test_hangul_jamo(self) -> None:
        """Hangul Jamo U+1100-U+1102 (ᄀᄁᄂ) must produce tokens."""
        tokens = _tokenize("ᄀᄁᄂ")
        assert len(tokens) > 0, "Hangul Jamo returned no tokens"

    def test_latin_still_works(self) -> None:
        """Existing Latin tokenization must remain intact."""
        tokens = _tokenize("hello world example test")
        assert len(tokens) > 0, "Latin tokenization regressed"
        assert "hello" in tokens
