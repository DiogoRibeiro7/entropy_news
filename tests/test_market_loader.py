from entropy_news.data import load_market_csv


def test_load_market_csv(tmp_path) -> None:
    content = "date,open,close\n2024-01-01,1.0,2.0\n2024-01-02,3.0,4.0\n"
    file = tmp_path / "market.csv"
    file.write_text(content)
    records = load_market_csv(str(file))
    assert len(records) == 2
    assert records[0].features == [1.0, 2.0]
