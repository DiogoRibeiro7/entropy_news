import pytest

from entropy_news.model import ConcatFusion, WeightedFusion

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - torch missing
    torch = None


def test_concat_fusion() -> None:
    fusion = ConcatFusion()
    if torch and hasattr(fusion, "forward"):
        text = torch.tensor([1.0, 2.0])
        market = torch.tensor([3.0])
        result = fusion(text, market)
        assert result.tolist() == [1.0, 2.0, 3.0]
    else:
        result = fusion([1.0, 2.0], [3.0])
        assert result == [1.0, 2.0, 3.0]


def test_weighted_fusion_scales_modalities() -> None:
    fusion = WeightedFusion(init_weights=(0.7, 0.3))
    if torch and hasattr(fusion, "forward"):
        text = torch.tensor([2.0])
        market = torch.tensor([4.0])
        result = fusion(text, market)
        assert pytest.approx(result.item()) == 0.7 * 2.0 + 0.3 * 4.0
    else:
        result = fusion([2.0], [4.0])
        assert result == [0.7 * 2.0 + 0.3 * 4.0]
