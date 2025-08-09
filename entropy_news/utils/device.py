"""Helpers for selecting compute devices and managing GPU features."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    import torch


def get_device() -> "torch.device":
    """Return the best available :mod:`torch` device.

    Returns:
        torch.device: ``cuda`` when available, otherwise ``cpu``.
    """

    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cuda_stream() -> "torch.cuda.Stream | None":
    """Return a new CUDA stream if available.

    Returns ``None`` when CUDA is unavailable or :mod:`torch` is missing.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        return None

    if not torch.cuda.is_available():
        return None
    return torch.cuda.Stream()


@contextmanager
def cuda_stream(stream: "torch.cuda.Stream | None") -> Iterator[None]:
    """Context manager that switches to ``stream`` when provided.

    Falls back to a no-op when CUDA is unavailable or ``stream`` is ``None``.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        stream = None

    if stream is None:
        with nullcontext():
            yield
    else:  # pragma: no cover - executed only when CUDA is present
        with torch.cuda.stream(stream):
            yield


@contextmanager
def autocast(use_amp: bool = True) -> Iterator[None]:
    """Context manager for automatic mixed precision.

    Args:
        use_amp: Enable AMP when ``True`` and supported.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        use_amp = False

    if not use_amp:
        with nullcontext():
            yield
        return

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type):
        yield
