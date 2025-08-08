"""Attention-related utilities."""

from __future__ import annotations

from typing import Any, Sequence


def plot_attention_weights(weights: Sequence[Sequence[float]], *, show: bool = False) -> Any:
    """Render a heatmap of attention ``weights``.

    The function uses matplotlib when available and returns the created figure.

    Args:
        weights: 2D attention matrix with shape ``(query, key)``.
        show: Whether to display the figure immediately.

    Returns:
        The matplotlib figure containing the heatmap.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    try:  # Lazy import to keep matplotlib optional
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - matplotlib missing
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    ax.imshow(weights, aspect="auto")
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    ax.set_title("Attention Weights")
    if show:  # pragma: no cover - interactive display
        plt.show()
    return fig
