"""Tests for device selection helper."""

import pytest

torch = pytest.importorskip("torch")
from entropy_news.utils import autocast, cuda_stream, get_cuda_stream, get_device


def test_get_device_cpu(monkeypatch) -> None:
    """Return CPU when CUDA is unavailable."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(get_device()) == "cpu"


def test_get_device_cuda(monkeypatch) -> None:
    """Return CUDA when available."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(get_device()) == "cuda"


def test_get_cuda_stream_unavailable(monkeypatch) -> None:
    """Return ``None`` when CUDA is unavailable."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_cuda_stream() is None


def test_cuda_stream_context(monkeypatch) -> None:
    """Use provided stream when available."""

    dummy_stream = object()

    def fake_stream(stream):  # type: ignore[override]
        assert stream is dummy_stream

        class Ctx:
            def __enter__(self) -> None:  # pragma: no cover - simple marker
                called.append("enter")

            def __exit__(self, *args) -> None:  # pragma: no cover - simple marker
                called.append("exit")

        return Ctx()

    called: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream)

    with cuda_stream(dummy_stream):
        called.append("body")

    assert called == ["body", "enter", "exit"]


def test_autocast(monkeypatch) -> None:
    """Use ``torch.autocast`` when AMP is enabled."""

    calls: list[str] = []

    class Ctx:
        def __enter__(self) -> None:  # pragma: no cover - simple marker
            calls.append("enter")

        def __exit__(self, *args) -> None:  # pragma: no cover - simple marker
            calls.append("exit")

    def fake_autocast(device_type: str) -> Ctx:  # type: ignore[override]
        calls.append(device_type)
        return Ctx()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with autocast():
        calls.append("body")

    assert calls == ["cuda", "body", "enter", "exit"]
