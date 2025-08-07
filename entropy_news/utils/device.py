"""Helpers for selecting compute devices."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    import torch


def get_device() -> torch.device:
    """Return the best available :mod:`torch` device.

    Returns:
        torch.device: ``cuda`` when available, otherwise ``cpu``.
    """

    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
