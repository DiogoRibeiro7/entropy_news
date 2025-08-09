import pytest

from entropy_news.utils import correlation, plot_correlation, rolling_correlation


def test_correlation_perfect_match() -> None:
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    assert correlation(x, y) == pytest.approx(1.0)


def test_rolling_correlation_identical_series() -> None:
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    result = rolling_correlation(x, y, window=3)
    assert len(result) == 3
    for val in result:
        assert val == pytest.approx(1.0)


def test_plot_correlation_returns_figure() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    values = [0.1, 0.2, 0.3]
    fig = plot_correlation(values)
    assert fig is not None
    plt.close(fig)
