# entropy_news/utils/metrics.py

import math

def perplexity(entropy: float) -> float:
    """Return the perplexity associated with a cross-entropy value.

    Args:
        entropy: Cross-entropy value.

    Returns:
        float: ``math.inf`` if ``entropy`` is infinite, otherwise ``e`` raised
        to ``entropy``.
    """
    if math.isinf(entropy):
        return float("inf")
    return math.exp(entropy)
