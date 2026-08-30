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
            "ENT": [1.0, 1.5, 2.0],
            "ENT_NEWS": [0.8, 1.2, 1.9],
            "ENT_MODEL": [0.2, 0.3, 0.1],
        }
    )


def test_compute_correlations() -> None:
    corr = compute_correlations(_sample_frame())
    assert set(corr.columns) == set(METRIC_COLUMNS)
    assert corr.loc["ENT", "ENT_NEWS"] == pytest.approx(0.987829)


def test_summarise_metrics_columns() -> None:
    summary = summarise_metrics(_sample_frame())
    assert list(summary.columns) == ["mean", "std", "min", "max"]
    assert "ENT" in summary.index


def test_build_report_contains_sections() -> None:
    report = build_report(_sample_frame())
    assert "Summary statistics" in report
    assert "Correlation matrix" in report


def test_filter_by_month_returns_subset() -> None:
    filtered = filter_by_month(_sample_frame(), ["2023-02"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["month"] == "2023-02"


def test_dashboard_accepts_nonpaper_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "baseline_entropy": [1.0, 1.1],
            "updated_entropy": [0.9, 1.0],
            "model_update_delta": [-0.1, -0.1],
        }
    )
    summary = summarise_metrics(frame)
    assert "baseline_entropy" in summary.index
