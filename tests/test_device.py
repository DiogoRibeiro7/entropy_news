"""Tests for device selection helper."""

import pytest

torch = pytest.importorskip("torch")
from entropy_news.utils import get_device


def test_get_device_cpu(monkeypatch) -> None:
    """Return CPU when CUDA is unavailable."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(get_device()) == "cpu"


def test_get_device_cuda(monkeypatch) -> None:
    """Return CUDA when available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(get_device()) == "cuda"
