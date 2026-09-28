from pathlib import Path

import pytest
from dataexcept import FileReadError, FileWriteError

from entropy_news.utils import load_texts, save_texts


def test_save_and_load_texts(tmp_path) -> None:
    """Ensure text helper writes and reads data correctly."""

    lines = ["foo", "bar", ""]
    out_file = tmp_path / "texts.txt"
    save_texts(lines, out_file)
    assert out_file.exists()
    assert load_texts(out_file) == ["foo", "bar"]


def test_save_texts_creates_dirs(tmp_path) -> None:
    """``save_texts`` should create parent directories when needed."""

    nested = tmp_path / "nested" / "deep" / "texts.txt"
    save_texts(["hello"], nested)
    assert nested.exists()


def test_load_texts_missing_file(tmp_path) -> None:
    """Missing input files should raise a clear ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError, match="not found"):
        load_texts(tmp_path / "missing.txt")


def test_load_texts_empty_file(tmp_path) -> None:
    """Empty files should raise a ``ValueError``."""

    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        load_texts(empty)


def test_load_texts_unreadable_path_retains_original_cause(tmp_path: Path) -> None:
    """A directory at the input path retains its I/O cause and source path."""
    source = tmp_path / "texts.txt"
    source.mkdir()

    with pytest.raises(FileReadError) as caught:
        load_texts(source)
    assert caught.value.path == str(source)
    assert isinstance(caught.value.original, IsADirectoryError)
    assert caught.value.__cause__ is caught.value.original


def test_load_texts_invalid_utf8_retains_decode_cause(tmp_path: Path) -> None:
    """Bad input encoding is reported with its original decode failure."""
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff")

    with pytest.raises(FileReadError) as caught:
        load_texts(source)
    assert caught.value.path == str(source)
    assert isinstance(caught.value.original, UnicodeDecodeError)


def test_save_texts_failed_directory_creation_retains_output_path(tmp_path: Path) -> None:
    """A blocked output directory is reported as a structured write failure."""
    blocker = tmp_path / "blocked"
    blocker.write_text("file", encoding="utf-8")
    output = blocker / "texts.txt"

    with pytest.raises(FileWriteError) as caught:
        save_texts(["hello"], output)
    assert caught.value.path == str(output)
    assert isinstance(caught.value.original, OSError)
    assert caught.value.__cause__ is caught.value.original
