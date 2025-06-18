from entropy_news.utils import load_texts, save_texts


def test_save_and_load_texts(tmp_path) -> None:
    """Ensure text helper writes and reads data correctly."""

    lines = ["foo", "bar", ""]
    out_file = tmp_path / "texts.txt"
    save_texts(lines, out_file)
    assert out_file.exists()
    assert load_texts(out_file) == ["foo", "bar"]
