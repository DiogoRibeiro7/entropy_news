"""Helpers for optimising inference deployments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - optional torch dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch missing
    torch = None
    nn = object  # type: ignore[misc]


def quantize_dynamic(model: nn.Module, *, dtype: torch.dtype | None = None) -> nn.Module:
    """Apply dynamic quantisation to ``model`` when torch is available."""

    if torch is None:  # pragma: no cover - guard for environments without torch
        raise RuntimeError("torch is required for quantisation")
    dtype = dtype or torch.qint8
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)  # type: ignore[arg-type]


def export_to_onnx(
    model: nn.Module,
    dummy_input: Iterable[int],
    path: str | Path,
    *,
    opset_version: int = 13,
) -> Path:
    """Export ``model`` to ONNX format using ``dummy_input`` for tracing."""

    if torch is None:  # pragma: no cover - guard for environments without torch
        raise RuntimeError("torch is required for ONNX export")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.tensor(list(dummy_input), dtype=torch.long).unsqueeze(0)
    torch.onnx.export(model, dummy, output_path.as_posix(), opset_version=opset_version, input_names=["tokens"], output_names=["logits"])
    return output_path
