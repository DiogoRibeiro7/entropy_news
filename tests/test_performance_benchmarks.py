from __future__ import annotations

import pytest

from entropy_news.utils import benchmark


@pytest.mark.performance
def test_benchmark_records_duration() -> None:
    calls: list[int] = []

    def workload() -> None:
        calls.append(1)

    result = benchmark(workload, repeats=3, warmup=1)
    assert result["time"] >= 0.0
    assert result["memory"] == 0.0
    assert len(calls) == 4
