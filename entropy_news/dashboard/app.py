"""Streamlit dashboard for visualising paper and diagnostic entropy outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - streamlit optional
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

import pandas as pd


PAPER_METRIC_COLUMNS: tuple[str, ...] = ("ENT", "ENT_NEWS", "ENT_MODEL")
DIAGNOSTIC_METRIC_COLUMNS: tuple[str, ...] = (
    "baseline_entropy",
    "updated_entropy",
    "model_update_delta",
)
# Backwards-compatible public name now points to the scientifically preferred set.
METRIC_COLUMNS = PAPER_METRIC_COLUMNS


def _select_metric_columns(
    df: pd.DataFrame, metrics: Iterable[str] | None = None
) -> list[str]:
    if metrics is not None:
        return [column for column in metrics if column in df.columns]
    paper = [column for column in PAPER_METRIC_COLUMNS if column in df.columns]
    if paper:
        return paper
    return [column for column in DIAGNOSTIC_METRIC_COLUMNS if column in df.columns]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(path)


def compute_correlations(
    df: pd.DataFrame, metrics: Iterable[str] | None = None
) -> pd.DataFrame:
    columns = _select_metric_columns(df, metrics)
    if not columns:
        raise ValueError("No matching entropy metrics available")
    return df[columns].corr().fillna(0.0)


def summarise_metrics(
    df: pd.DataFrame, metrics: Iterable[str] | None = None
) -> pd.DataFrame:
    columns = _select_metric_columns(df, metrics)
    if not columns:
        raise ValueError("No matching entropy metrics available")
    summary = df[columns].agg(["mean", "std", "min", "max"]).transpose()
    summary.index.name = "metric"
    return summary


def build_report(df: pd.DataFrame, metrics: Iterable[str] | None = None) -> str:
    summary = summarise_metrics(df, metrics).round(4)
    correlations = compute_correlations(df, metrics).round(4)
    return "\n".join(
        [
            "Entropy Forecast Report",
            "=======================",
            "",
            "Summary statistics:",
            summary.to_string(),
            "",
            "Correlation matrix:",
            correlations.to_string(),
            "",
        ]
    )


def create_heatmap(correlations: pd.DataFrame):
    try:  # pragma: no cover
        import altair as alt
    except Exception:  # pragma: no cover
        return None
    data = correlations.reset_index().melt(
        "index", var_name="metric", value_name="correlation"
    )
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("metric:O", title="Metric"),
            y=alt.Y("index:O", title="Metric"),
            color=alt.Color(
                "correlation:Q", scale=alt.Scale(domain=(-1, 1), scheme="redblue")
            ),
            tooltip=["index", "metric", alt.Tooltip("correlation", format=".2f")],
        )
        .properties(height=300)
    )


def filter_by_month(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    if "month" not in df.columns or not months:
        return df
    return df[df["month"].isin(months)]


def main(csv_path: str = "output/paper_reproduction/paper_entropy_results.csv") -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to run the dashboard")
    st.title("Entropy News Dashboard")
    st.write("Paper reproduction metrics are preferred when available.")
    csv_source = st.sidebar.text_input("Results CSV", value=str(csv_path))
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    try:
        df = pd.read_csv(uploaded) if uploaded is not None else load_results(csv_source)
    except FileNotFoundError as exc:  # pragma: no cover
        st.error(str(exc))
        return
    metrics = _select_metric_columns(df)
    if not metrics:
        st.warning("No recognised entropy metrics found in the provided dataset.")
        return
    month_options = df["month"].astype(str).tolist() if "month" in df.columns else []
    unique_months = sorted(set(month_options))
    selected_months = (
        st.sidebar.multiselect("Months", unique_months, default=unique_months)
        if unique_months
        else []
    )
    filtered = filter_by_month(df, selected_months)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return
    st.dataframe(filtered)
    if "month" in filtered.columns:
        st.line_chart(filtered.set_index("month")[metrics])
    st.subheader("Summary statistics")
    st.dataframe(summarise_metrics(filtered, metrics).style.format("{:.4f}"))
    correlations = compute_correlations(filtered, metrics)
    st.subheader("Correlation heatmap")
    heatmap = create_heatmap(correlations)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
    else:  # pragma: no cover
        st.dataframe(correlations.style.format("{:.2f}"))
    st.download_button(
        "Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="entropy_forecasts.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download summary report",
        build_report(filtered, metrics).encode("utf-8"),
        file_name="entropy_dashboard_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
