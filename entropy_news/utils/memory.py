"""Memory usage profiling utilities."""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def measure_peak_memory(func: Callable[..., T], *args, **kwargs) -> tuple[T, int]:
    """Run ``func`` and return its result with peak memory usage.

    Args:
        func: Callable to execute.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        tuple containing the callable result and peak memory usage in bytes.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, peak
