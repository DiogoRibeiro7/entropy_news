"""Utilities for loading numerical market data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import json
from typing import List, Sequence
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class MarketRecord:
    """Container for a single market data row."""

    date: str
    features: List[float]


def load_market_csv(file_path: str, columns: Sequence[str] | None = None) -> List[MarketRecord]:
    """Load market data from ``file_path``.

    Args:
        file_path: CSV file path containing a ``date`` column and numeric features.
        columns: Optional list of feature column names to load. All columns except
            ``date`` are used when omitted.
    """
    records: List[MarketRecord] = []
    with open(file_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return records
        selected = [c for c in reader.fieldnames if c != "date"] if columns is None else list(columns)
        for row in reader:
            features = [float(row[col]) for col in selected if col in row and row[col]]
            records.append(MarketRecord(date=row.get("date", ""), features=features))
    return records


def fetch_yahoo_history(
    symbol: str,
    start: str,
    end: str,
    columns: Sequence[str] | None = None,
) -> List[MarketRecord]:
    """Download daily price data from Yahoo Finance.

    Args:
        symbol: Ticker symbol to query.
        start: Start date in ``YYYY-MM-DD`` format.
        end: End date in ``YYYY-MM-DD`` format.
        columns: Optional list of feature column names to load. All columns
            except ``Date`` are used when omitted.

    Raises:
        ConnectionError: If the Yahoo request fails.
    """

    period1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        f"{symbol}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    try:
        with urlopen(url) as response:
            data = response.read().decode("utf-8")
    except URLError as err:  # pragma: no cover - network failure
        raise ConnectionError("failed to fetch data") from err

    reader = csv.DictReader(StringIO(data))
    if reader.fieldnames is None:
        return []
    selected = (
        [c for c in reader.fieldnames if c != "Date"]
        if columns is None
        else list(columns)
    )
    records: List[MarketRecord] = []
    for row in reader:
        features = [float(row[col]) for col in selected if row.get(col)]
        records.append(MarketRecord(date=row.get("Date", ""), features=features))
    return records


def fetch_alpha_vantage_history(
    symbol: str,
    api_key: str,
    outputsize: str = "compact",
    columns: Sequence[str] | None = None,
) -> List[MarketRecord]:
    """Download daily price data from Alpha Vantage.

    Args:
        symbol: Ticker symbol to query.
        api_key: Alpha Vantage API key.
        outputsize: "compact" or "full" data range.
        columns: Optional list of feature column names to load. Defaults to the
            adjusted close price when omitted.

    Raises:
        ConnectionError: If the Alpha Vantage request fails.
        ValueError: If the response is malformed.
    """

    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&apikey={api_key}&outputsize={outputsize}&datatype=json"
    )
    try:
        with urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as err:  # pragma: no cover - network failure
        raise ConnectionError("failed to fetch data") from err

    series_key = "Time Series (Daily)"
    if series_key not in data:
        raise ValueError("invalid response")
    timeseries = data[series_key]
    selected = ["5. adjusted close"] if columns is None else list(columns)

    records: List[MarketRecord] = []
    for date_str in sorted(timeseries):
        row = timeseries[date_str]
        features = [float(row[col]) for col in selected if col in row]
        records.append(MarketRecord(date=date_str, features=features))
    return records
