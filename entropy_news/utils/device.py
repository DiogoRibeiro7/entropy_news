"""Helpers for selecting compute devices and managing GPU features."""

from __future__ import annotations

from contextlib import contextmanager
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
def _enter_context(ctx: "object") -> Iterator[None]:
    """Enter ``ctx`` and guarantee a matching ``__exit__`` call."""

    ctx.__enter__()
    try:
        yield
    except BaseException as exc:  # pragma: no cover - passthrough for tests
        if not ctx.__exit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        ctx.__exit__(None, None, None)


@contextmanager
def cuda_stream(stream: "torch.cuda.Stream | None") -> Iterator[None]:
    """Context manager that switches to ``stream`` while the block runs.

    Falls back to a no-op when CUDA is unavailable or ``stream`` is ``None``.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency
        stream = None

    if stream is None:
        yield
        return

    with _enter_context(torch.cuda.stream(stream)):
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
        yield
        return

    if not torch.cuda.is_available():
        device_type = "cpu"
    else:
        device_type = "cuda"

    autocast_factory = getattr(torch, "autocast", None)
    if autocast_factory is None:  # pragma: no cover - older PyTorch versions
        yield
        return

    with _enter_context(autocast_factory(device_type=device_type)):
        yield
