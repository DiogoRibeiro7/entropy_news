import pytest

matplotlib = pytest.importorskip("matplotlib")

from entropy_news.utils import plot_attention_weights


def test_plot_attention_weights_returns_figure() -> None:
    weights = [[0.1, 0.9], [0.5, 0.5]]
    fig = plot_attention_weights(weights)
    assert fig is not None
