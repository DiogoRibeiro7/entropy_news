"""Stress-suite coverage for synthetic reliability scenarios."""

from __future__ import annotations

import math

import pytest

from entropy_news.utils.stress import (
    StressReport,
    StressScenario,
    run_stress_matrix,
    run_stress_scenario,
    simulate_burst_workload,
)

pytestmark = pytest.mark.stress


def _lightweight_workload(batch_size: int) -> None:
    """CPU-bound helper to provide deterministic work for the stress harness."""

    total = 0.0
    for index in range(batch_size * 2):
        total += math.sin(index) * math.cos(index / 2)
    if total < -1e12:
        raise RuntimeError("workload underflow detected")


def test_run_stress_scenario_collects_expected_metrics() -> None:
    """Running a scenario returns throughput statistics and latency samples."""

    scenario = StressScenario(
        name="unit-stress",
        iterations=5,
        batch_size=8,
        workload=_lightweight_workload,
        warmup_iterations=1,
    )
    report = run_stress_scenario(scenario)

    assert isinstance(report, StressReport)
    assert report.total_seconds > 0
    assert len(report.iteration_durations) == scenario.iterations
    assert report.throughput_batches_per_second > 0
    assert report.throughput_items_per_second > report.throughput_batches_per_second
    assert report.max_iteration_seconds >= report.mean_iteration_seconds
    assert set(report.to_dict()) == {
        "iterations",
        "total_seconds",
        "throughput_batches_per_second",
        "throughput_items_per_second",
        "mean_iteration_seconds",
        "max_iteration_seconds",
    }


def test_run_stress_matrix_executes_all_scenarios() -> None:
    """The matrix runner evaluates every provided scenario in sequence."""

    scenarios = (
        StressScenario("first", iterations=2, batch_size=4, workload=_lightweight_workload),
        StressScenario("second", iterations=3, batch_size=6, workload=_lightweight_workload),
    )
    reports = run_stress_matrix(scenarios)

    assert len(reports) == 2
    assert all(report.total_seconds > 0 for report in reports)


def test_simulate_burst_workload_is_side_effect_free() -> None:
    """The synthetic burst workload completes without raising errors."""

    assert simulate_burst_workload(4) is None
