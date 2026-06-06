"""ZIP build input — ``indx <archive.zip>`` / ``DirectoryPipeline().run("corpus.zip", …)``.

A ``.zip`` is accepted in place of a directory: it is extracted into a temp dir that becomes
the walk root, the manifest's ``source_root`` keeps pointing at the archive, the temp dir is
cleaned up afterwards, and unsafe (zip-slip) members are rejected. All offline (plaintext /
hash / jsonl) so the suite stays network-free. Mirrors pipeline/walk.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from indx.pipeline import BuildPlan, DirectoryPipeline
from indx.utils.zip_input import ZipInputError, extract_zip, is_zip_input

OFFLINE = {"parser": "plaintext", "llm": "none", "embedder": "hash", "store": "jsonl"}


def _corpus_zip(dest: Path) -> Path:
    """A small two-file corpus packed into a .zip; returns the archive path."""
    zip_path = dest / "corpus.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("notes/onboarding.md", "# Onboarding\n\nWelcome aboard. Alpha beta gamma.\n")
        zf.writestr("policy.txt", "Time off policy: ask your lead.\n")
    return zip_path


def _live_temp_dirs() -> list[Path]:
    return list(Path("/tmp").glob("indx-zip-*")) if Path("/tmp").is_dir() else []


def test_is_zip_input(tmp_path: Path) -> None:
    z = _corpus_zip(tmp_path)
    assert is_zip_input(z)
    assert not is_zip_input(tmp_path)  # a directory
    txt = tmp_path / "a.txt"
    txt.write_text("hi")
    assert not is_zip_input(txt)


def test_build_from_zip_produces_a_space(tmp_path: Path) -> None:
    z = _corpus_zip(tmp_path)
    space = DirectoryPipeline(**OFFLINE).run(z, tmp_path / "out")  # type: ignore[arg-type]
    assert space.stats.documents == 2
    # source_root stays the archive (portable), not the throwaway temp dir.
    assert space.manifest.source_root == str(z)
    # provenance survived: every chunk carries a resolvable source path from inside the zip.
    paths = {h.source.path for h in space.search("onboarding", k=3) if h.source}
    assert any(p.endswith("onboarding.md") for p in paths)


def test_zip_temp_dir_is_cleaned_up(tmp_path: Path) -> None:
    before = set(_live_temp_dirs())
    z = _corpus_zip(tmp_path)
    DirectoryPipeline(**OFFLINE).run(z, tmp_path / "out")  # type: ignore[arg-type]
    assert set(_live_temp_dirs()) == before  # no leaked extraction dirs


def test_plan_from_zip_reports_archive_as_root(tmp_path: Path) -> None:
    before = set(_live_temp_dirs())
    z = _corpus_zip(tmp_path)
    plan = DirectoryPipeline(**OFFLINE).plan(z)  # type: ignore[arg-type]
    assert isinstance(plan, BuildPlan)
    assert plan.root == z  # the archive, not the temp extraction dir
    assert len(plan.documents) == 2
    assert set(_live_temp_dirs()) == before  # dry-run cleans up too


def test_zip_slip_member_is_rejected(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escape.txt", "pwned")
    before = set(_live_temp_dirs())
    with pytest.raises(ZipInputError, match="zip-slip"):
        extract_zip(evil)
    assert set(_live_temp_dirs()) == before  # the partial extraction dir is removed on failure


def test_corrupt_zip_raises_zipinputerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a real zip")
    with pytest.raises(ZipInputError):
        extract_zip(bad)


def test_non_zip_file_input_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "lonely.txt"
    f.write_text("just a file")
    with pytest.raises(NotADirectoryError):
        DirectoryPipeline(**OFFLINE).run(f, tmp_path / "out")  # type: ignore[arg-type]
