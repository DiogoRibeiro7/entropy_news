import pytest

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
