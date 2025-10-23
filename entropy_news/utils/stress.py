"""Utilities to orchestrate synthetic stress scenarios for reliability checks."""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

Workload = Callable[[int], None]


@dataclass(slots=True)
class StressScenario:
    """Configuration for a synthetic stress workload.

    Attributes:
        name: Human readable identifier for the scenario.
        iterations: Number of measured iterations to execute.
        batch_size: Workload batch size dispatched on each iteration.
        workload: Callable that performs the synthetic work for a given batch
            size. The callable is expected to be CPU bound so that stress runs
            remain portable across CI runners.
        warmup_iterations: Optional number of warmup iterations executed before
            measurements begin. Warmups stabilise caches and JIT compilers so
            collected metrics are representative.
    """

    name: str
    iterations: int
    batch_size: int
    workload: Workload
    warmup_iterations: int = 1

    def __post_init__(self) -> None:
        """Validate scenario configuration eagerly."""

        if self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")


@dataclass(slots=True)
class StressReport:
    """Collected metrics from executing a :class:`StressScenario`."""

    scenario: StressScenario
    iteration_durations: Sequence[float]
    total_seconds: float

    @property
    def throughput_batches_per_second(self) -> float:
        """Compute achieved throughput in batches per second."""

        if self.total_seconds == 0:
            return 0.0
        return len(self.iteration_durations) / self.total_seconds

    @property
    def throughput_items_per_second(self) -> float:
        """Compute throughput translated to processed items per second."""

        if self.total_seconds == 0:
            return 0.0
        return (
            len(self.iteration_durations)
            * self.scenario.batch_size
            / self.total_seconds
        )

    @property
    def mean_iteration_seconds(self) -> float:
        """Return the mean iteration latency in seconds."""

        return statistics.fmean(self.iteration_durations)

    @property
    def max_iteration_seconds(self) -> float:
        """Return the slowest iteration latency."""

        return max(self.iteration_durations)

    def to_dict(self) -> Dict[str, float]:
        """Materialise the report metrics as a mapping."""

        return {
            "iterations": float(len(self.iteration_durations)),
            "total_seconds": self.total_seconds,
            "throughput_batches_per_second": self.throughput_batches_per_second,
            "throughput_items_per_second": self.throughput_items_per_second,
            "mean_iteration_seconds": self.mean_iteration_seconds,
            "max_iteration_seconds": self.max_iteration_seconds,
        }


def run_stress_scenario(scenario: StressScenario) -> StressReport:
    """Execute a stress scenario and collect timing metrics.

    Args:
        scenario: The scenario configuration to execute.

    Returns:
        A :class:`StressReport` containing iteration-level latency metrics and
        aggregated throughput figures.
    """

    for _ in range(scenario.warmup_iterations):
        scenario.workload(scenario.batch_size)

    iteration_durations: List[float] = []
    start = time.perf_counter()
    for _ in range(scenario.iterations):
        iter_start = time.perf_counter()
        scenario.workload(scenario.batch_size)
        iteration_durations.append(time.perf_counter() - iter_start)
    total_seconds = time.perf_counter() - start
    return StressReport(
        scenario=scenario,
        iteration_durations=tuple(iteration_durations),
        total_seconds=total_seconds,
    )


def run_stress_matrix(scenarios: Iterable[StressScenario]) -> List[StressReport]:
    """Execute a collection of stress scenarios sequentially."""

    return [run_stress_scenario(scenario) for scenario in scenarios]


def simulate_burst_workload(batch_size: int) -> None:
    """Synthetic CPU-bound workload that mimics bursty token processing."""

    _ = 0.0
    for index in range(batch_size * 4):
        jitter = 1.0 + 0.15 * math.sin(index)
        _ += math.tanh(random.random() * jitter)
    if _ < 0:
        # This branch is never taken but ensures the accumulator is not
        # optimised away by aggressive interpreters.
        raise RuntimeError("stress workload failed to accumulate")
