from io import BytesIO
import entropy_news.data.market as market
from entropy_news.data import fetch_yahoo_history


class DummyResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_fetch_yahoo_history(monkeypatch) -> None:
    csv_data = "Date,Open,Close\n2024-01-01,1,2\n2024-01-02,3,4\n"

    def fake_urlopen(url):
        return DummyResponse(csv_data.encode("utf-8"))

    monkeypatch.setattr(market, "urlopen", fake_urlopen)
    records = fetch_yahoo_history("AAPL", "2024-01-01", "2024-01-03", columns=["Open", "Close"])
    assert len(records) == 2
    assert records[0].date == "2024-01-01"
    assert records[0].features == [1.0, 2.0]
