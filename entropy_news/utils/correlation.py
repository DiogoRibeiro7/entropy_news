"""Correlation utilities for research analysis."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import StatisticsError, correlation as _corr


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Pearson correlation between two sequences.

    Args:
        x: First numeric series.
        y: Second numeric series of equal length.

    Raises:
        ValueError: If the series are empty or lengths differ.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("series must have the same non-zero length")
    return float(_corr(x, y))


def rolling_correlation(
    x: Sequence[float],
    y: Sequence[float],
    window: int,
) -> list[float]:
    """Compute rolling window correlation between two sequences.

    Args:
        x: First numeric series.
        y: Second numeric series of equal length.
        window: Number of observations per correlation computation.

    Returns:
        List of correlation coefficients, one per completed window.

    Raises:
        ValueError: If ``window`` is not positive or the series are invalid.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(x) != len(y):
        raise ValueError("series must have equal length")
    if len(x) < window:
        raise ValueError("series length must be >= window")

    arr_x = [float(v) for v in x]
    arr_y = [float(v) for v in y]
    result: list[float] = []
    for i in range(window, len(arr_x) + 1):
        seg_x = arr_x[i - window : i]
        seg_y = arr_y[i - window : i]
        try:
            result.append(float(_corr(seg_x, seg_y)))
        except StatisticsError:
            result.append(float("nan"))
    return result


def plot_correlation(values: Sequence[float], *, show: bool = False):
    """Plot correlation coefficients over time.

    Args:
        values: Sequence of correlation coefficients to visualize.
        show: Whether to display the plot immediately. Defaults to ``False``.

    Returns:
        The ``matplotlib`` figure containing the plot.

    Raises:
        ImportError: If ``matplotlib`` is not available.
    """
    try:  # Import lazily to avoid mandatory dependency
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    ax.plot(range(len(values)), list(values))
    ax.set_xlabel("index")
    ax.set_ylabel("correlation")
    ax.set_ylim(-1, 1)
    if show:
        plt.show()
    return fig

