"""Utilities for loading numerical market data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Sequence


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
