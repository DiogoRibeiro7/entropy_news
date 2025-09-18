from __future__ import annotations

import pandas as pd
import pytest

from entropy_news.dashboard.app import (
    METRIC_COLUMNS,
    build_report,
    compute_correlations,
    filter_by_month,
    summarise_metrics,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["2023-01", "2023-02", "2023-03"],
            "entropy": [1.0, 1.5, 2.0],
            "entropy_news": [0.8, 1.2, 1.9],
            "entropy_model": [0.6, 0.9, 1.4],
        }
    )


def test_compute_correlations() -> None:
    df = _sample_frame()
    corr = compute_correlations(df)
    assert set(corr.columns) == set(METRIC_COLUMNS)
    assert corr.loc["entropy", "entropy_news"] == pytest.approx(0.987829)


def test_summarise_metrics_columns() -> None:
    df = _sample_frame()
    summary = summarise_metrics(df)
    assert list(summary.columns) == ["mean", "std", "min", "max"]
    assert "entropy" in summary.index


def test_build_report_contains_sections() -> None:
    df = _sample_frame()
    report = build_report(df)
    assert "Summary statistics" in report
    assert "Correlation matrix" in report


def test_filter_by_month_returns_subset() -> None:
    df = _sample_frame()
    filtered = filter_by_month(df, ["2023-02"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["month"] == "2023-02"
