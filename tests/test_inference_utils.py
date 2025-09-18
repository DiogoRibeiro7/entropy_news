from __future__ import annotations

from pathlib import Path

import torch

from entropy_news.model.inference import export_to_onnx, quantize_dynamic


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.float())


def test_quantize_dynamic(monkeypatch) -> None:
    model = TinyModel()

    called = {}

    def fake_quantize(mod, layers, dtype):  # type: ignore[override]
        called["dtype"] = dtype
        return mod

    monkeypatch.setattr(torch.quantization, "quantize_dynamic", fake_quantize)
    quantized = quantize_dynamic(model)
    assert called["dtype"] == torch.qint8
    assert quantized is model


def test_export_to_onnx(monkeypatch, tmp_path: Path) -> None:
    model = TinyModel()
    captured: dict[str, Path] = {}

    def fake_export(mod, dummy, path, **kwargs):  # type: ignore[override]
        captured["path"] = Path(path)
        captured["dummy"] = dummy

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = export_to_onnx(model, [0, 1, 2, 3], tmp_path / "model.onnx")
    assert output == tmp_path / "model.onnx"
    assert captured["path"] == output
    assert captured["dummy"].shape[1] == 4
