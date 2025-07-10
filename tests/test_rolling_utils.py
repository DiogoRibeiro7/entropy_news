"""Tests for rolling forecast utility functions."""

from entropy_news.rolling_train_forecast import (
    load_texts_for_month,
    load_texts_for_months,
)


def test_load_texts_for_month(tmp_path) -> None:
    """Single-month loader should read correct lines."""

    base = tmp_path
    file_path = base / "news_2023-01.txt"
    file_path.write_text("a\nb\n\n")

    result = load_texts_for_month("2023-01", str(base))

    assert result == ["a", "b"]


def test_load_texts_for_months(tmp_path) -> None:
    """Multi-month helper concatenates all monthly texts."""

    base = tmp_path
    (base / "news_2023-01.txt").write_text("a\n")
    (base / "news_2023-02.txt").write_text("b\nc\n")

    result = load_texts_for_months(["2023-01", "2023-02"], str(base))

    assert result == ["a", "b", "c"]
