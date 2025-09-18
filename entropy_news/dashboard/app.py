"""Streamlit dashboard for visualising entropy forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - streamlit optional
    import streamlit as st
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    st = None

import pandas as pd


METRIC_COLUMNS: tuple[str, ...] = ("entropy", "entropy_news", "entropy_model")


def _select_metric_columns(df: pd.DataFrame, metrics: Iterable[str] | None = None) -> list[str]:
    """Return the intersection between ``metrics`` and the dataframe columns."""

    candidates = list(metrics) if metrics is not None else list(METRIC_COLUMNS)
    return [column for column in candidates if column in df.columns]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load rolling forecast CSV results."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(path)


def compute_correlations(
    df: pd.DataFrame, metrics: Iterable[str] | None = None
) -> pd.DataFrame:
    """Compute correlation coefficients between entropy metrics."""

    columns = _select_metric_columns(df, metrics)
    if not columns:
        raise ValueError("No matching metrics available for correlation analysis")
    return df[columns].corr().fillna(0.0)


def summarise_metrics(
    df: pd.DataFrame, metrics: Iterable[str] | None = None
) -> pd.DataFrame:
    """Produce descriptive statistics for dashboard metrics."""

    columns = _select_metric_columns(df, metrics)
    if not columns:
        raise ValueError("No matching metrics available for summary statistics")
    summary = df[columns].agg(["mean", "std", "min", "max"]).transpose()
    summary.index.name = "metric"
    return summary


def build_report(df: pd.DataFrame, metrics: Iterable[str] | None = None) -> str:
    """Generate a textual report summarising key dashboard insights."""

    summary = summarise_metrics(df, metrics).round(4)
    correlations = compute_correlations(df, metrics).round(4)
    lines = [
        "Entropy Forecast Report",
        "======================",
        "",
        "Summary statistics:",
        summary.to_string(),
        "",
        "Correlation matrix:",
        correlations.to_string(),
        "",
    ]
    return "\n".join(lines)


def create_heatmap(correlations: pd.DataFrame):
    """Return an Altair heatmap for ``correlations`` when Altair is available."""

    try:  # pragma: no cover - optional dependency
        import altair as alt
    except Exception:  # pragma: no cover
        return None

    data = correlations.reset_index().melt(
        "index", var_name="metric", value_name="correlation"
    )
    chart = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("metric:O", title="Metric"),
            y=alt.Y("index:O", title="Metric"),
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(domain=(-1, 1), scheme="redblue"),
            ),
            tooltip=[
                alt.Tooltip("index", title="Row"),
                alt.Tooltip("metric", title="Column"),
                alt.Tooltip("correlation", title="Correlation", format=".2f"),
            ],
        )
        .properties(height=300)
    )
    return chart


def filter_by_month(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Filter ``df`` to rows whose ``month`` is in ``months``."""

    if "month" not in df.columns or not months:
        return df
    return df[df["month"].isin(months)]


def main(csv_path: str = "output/rolling_forecast_results.csv") -> None:
    """Render the dashboard."""

    if st is None:  # pragma: no cover - streamlit not installed during tests
        raise RuntimeError("Streamlit is required to run the dashboard")

    st.title("Entropy News Dashboard")
    st.write("Interactive view of rolling entropy forecasts.")

    st.sidebar.header("Data source")
    csv_source = st.sidebar.text_input("Results CSV", value=str(csv_path))
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")

    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            df = load_results(csv_source)
    except FileNotFoundError as exc:  # pragma: no cover - UI feedback
        st.error(str(exc))
        return

    metrics = _select_metric_columns(df)
    if not metrics:
        st.warning("No entropy metrics found in the provided dataset.")
        return

    st.sidebar.header("Filters")
    month_options = (
        df["month"].astype(str).tolist() if "month" in df.columns else []
    )
    unique_months = sorted(set(month_options))
    if unique_months:
        selected_months = st.sidebar.multiselect(
            "Months", options=unique_months, default=unique_months
        )
    else:
        selected_months = []

    filtered = filter_by_month(df, selected_months)
    if filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    st.dataframe(filtered)
    st.line_chart(filtered.set_index("month")[metrics])

    summary = summarise_metrics(filtered, metrics)
    st.subheader("Summary statistics")
    st.dataframe(summary.style.format("{:.4f}"))

    correlations = compute_correlations(filtered, metrics)
    st.subheader("Correlation heatmap")
    heatmap = create_heatmap(correlations)
    if heatmap is not None:
        st.altair_chart(heatmap, use_container_width=True)
    else:  # pragma: no cover - altair missing
        st.dataframe(correlations.style.format("{:.2f}"))

    st.download_button(
        "Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="entropy_forecasts.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download correlation matrix (CSV)",
        correlations.to_csv().encode("utf-8"),
        file_name="entropy_correlations.csv",
        mime="text/csv",
    )

    report_text = build_report(filtered, metrics)
    st.download_button(
        "Download summary report",
        report_text.encode("utf-8"),
        file_name="entropy_dashboard_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
