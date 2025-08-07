from entropy_news.utils import measure_peak_memory


def _allocate() -> list[int]:
    """Return a list to allocate memory."""
    return [0] * 1000


def test_measure_peak_memory_returns_usage() -> None:
    """Measure memory usage of a simple allocation."""
    data, peak = measure_peak_memory(_allocate)
    assert isinstance(data, list)
    assert peak > 0
