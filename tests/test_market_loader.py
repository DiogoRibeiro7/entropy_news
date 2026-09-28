from pathlib import Path

import pytest
from dataexcept import DataLoadingError

from entropy_news.data import load_market_csv


def test_load_market_csv(tmp_path) -> None:
    content = "date,open,close\n2024-01-01,1.0,2.0\n2024-01-02,3.0,4.0\n"
    file = tmp_path / "market.csv"
    file.write_text(content)
    records = load_market_csv(str(file))
    assert len(records) == 2
    assert records[0].features == [1.0, 2.0]


def test_load_market_csv_missing_file_preserves_source(tmp_path: Path) -> None:
    """Market input failures expose the source and original file error."""
    source = tmp_path / "missing.csv"
    with pytest.raises(DataLoadingError) as caught:
        load_market_csv(str(source))
    assert caught.value.source == str(source)
    assert isinstance(caught.value.original, FileNotFoundError)
    assert caught.value.__cause__ is caught.value.original


def test_load_market_csv_invalid_numeric_value_stays_a_validation_error(tmp_path: Path) -> None:
    """Invalid market values retain their existing conversion error."""
    source = tmp_path / "invalid.csv"
    source.write_text("date,close\n2024-01-01,not-a-price\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not-a-price"):
        load_market_csv(str(source))
