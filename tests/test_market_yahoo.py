from io import BytesIO
from urllib.error import URLError

import pytest
from dataexcept import ServiceConnectionError

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


def test_fetch_yahoo_connection_failure_preserves_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Yahoo request retains the provider and original network error."""
    def fail(_url: str) -> None:
        """Simulate an unavailable provider."""
        raise URLError("offline")

    monkeypatch.setattr(market, "urlopen", fail)
    with pytest.raises(ServiceConnectionError) as caught:
        fetch_yahoo_history("AAPL", "2024-01-01", "2024-01-03")
    assert caught.value.service_name == "Yahoo Finance"
    assert isinstance(caught.value.original_exception, URLError)
    assert caught.value.__cause__ is caught.value.original_exception
