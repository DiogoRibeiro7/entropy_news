import types
import pytest

import entropy_news
from entropy_news.utils import metrics, logger


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("perplexity", metrics.perplexity),
        ("setup_logger", logger.setup_logger),
    ],
)
def test_module_getattr_valid(attr, expected) -> None:
    """Check valid names exposed by :mod:`entropy_news`."""
    obj = getattr(entropy_news, attr)
    assert isinstance(obj, types.FunctionType)
    assert obj is expected


def test_module_getattr_invalid() -> None:
    """Unknown names should raise ``AttributeError``."""
    with pytest.raises(AttributeError):
        getattr(entropy_news, "does_not_exist")
