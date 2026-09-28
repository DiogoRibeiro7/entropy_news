from io import BytesIO
import json
from urllib.error import URLError

import pytest
from dataexcept import ServiceConnectionError

import entropy_news.data.market as market
from entropy_news.data import fetch_alpha_vantage_history


class DummyResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_fetch_alpha_vantage_history(monkeypatch) -> None:
    json_data = {
        "Time Series (Daily)": {
            "2024-01-02": {"5. adjusted close": "2"},
            "2024-01-01": {"5. adjusted close": "1"},
        }
    }

    def fake_urlopen(url):
        return DummyResponse(json.dumps(json_data).encode("utf-8"))

    monkeypatch.setattr(market, "urlopen", fake_urlopen)
    records = fetch_alpha_vantage_history("AAPL", "key", columns=["5. adjusted close"])
    assert len(records) == 2
    assert records[0].date == "2024-01-01"
    assert records[0].features == [1.0]


def test_fetch_alpha_vantage_connection_failure_preserves_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Alpha Vantage request retains the provider and network cause."""
    def fail(_url: str) -> None:
        """Simulate an unavailable provider."""
        raise URLError("offline")

    monkeypatch.setattr(market, "urlopen", fail)
    with pytest.raises(ServiceConnectionError) as caught:
        fetch_alpha_vantage_history("AAPL", "example-key")
    assert caught.value.service_name == "Alpha Vantage"
    assert isinstance(caught.value.original_exception, URLError)
    assert caught.value.__cause__ is caught.value.original_exception
