import types

import entropy_news
from entropy_news.utils import metrics, logger


def test_module_exports() -> None:
    """Ensure package-level names refer to utility functions."""
    assert isinstance(entropy_news.perplexity, types.FunctionType)
    assert entropy_news.perplexity is metrics.perplexity
    assert isinstance(entropy_news.setup_logger, types.FunctionType)
    assert entropy_news.setup_logger is logger.setup_logger
