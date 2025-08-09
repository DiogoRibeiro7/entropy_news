import pytest

from entropy_news.utils import benchmark


def test_benchmark_reports_time_and_memory():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    stats = benchmark(model, x, repeats=2, measure_memory=True)
    assert stats["time"] > 0
    assert stats["memory"] >= 0
