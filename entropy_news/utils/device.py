"""Helpers for selecting compute devices."""

from __future__ import annotations


def get_device() -> "torch.device":
    """Return the default ``torch`` device.

    Prefers CUDA when available, falling back to CPU otherwise.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
