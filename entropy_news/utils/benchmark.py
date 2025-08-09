"""Utilities for benchmarking model execution speed and memory."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict

try:  # pragma: no cover - optional torch import
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore

from .memory import measure_peak_memory


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 10,
    warmup: int = 1,
    measure_memory: bool = False,
    **kwargs: Any,
) -> Dict[str, float]:
    """Benchmark a callable's execution time and optional memory usage.

    The callable is executed ``warmup`` times without measurement, followed by
    ``repeats`` timed iterations. If ``measure_memory`` is ``True``, peak memory
    consumption for the timed iterations is also recorded.

    Args:
        func: Callable to benchmark.
        *args: Positional arguments forwarded to ``func``.
        repeats: Number of timed iterations to execute.
        warmup: Number of warm-up iterations run before timing.
        measure_memory: Whether to measure peak memory usage in bytes.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Dictionary containing average execution time per iteration under key
        ``"time"`` (seconds) and peak memory usage under key ``"memory"``
        (bytes). When ``measure_memory`` is ``False``, ``"memory"`` will be ``0``.
    """

    for _ in range(warmup):
        func(*args, **kwargs)

    def _run() -> float:
        start = perf_counter()
        for _ in range(repeats):
            func(*args, **kwargs)
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - CUDA
            torch.cuda.synchronize()
        return perf_counter() - start

    if measure_memory:
        duration, peak = measure_peak_memory(_run)
    else:
        duration = _run()
        peak = 0

    return {"time": duration / repeats, "memory": float(peak)}

